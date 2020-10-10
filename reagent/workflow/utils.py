#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

import pytorch_lightning as pl
import reagent.types as rlt

# pyre-fixme[21]: Could not find `petastorm`.
from petastorm import make_batch_reader

# pyre-fixme[21]: Could not find module `petastorm.pytorch`.
# pyre-fixme[21]: Could not find module `petastorm.pytorch`.
from petastorm.pytorch import DataLoader, decimal_friendly_collate
from reagent.core.tracker import Observer
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.evaluator import Evaluator
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.torch_utils import dict_to_tensor
from reagent.training import RLTrainer, SACTrainer, StoppingEpochCallback, TD3Trainer
from reagent.workflow_utils.iterators import DataLoaderWrapper, EpochIterator

from .spark_utils import get_spark_session
from .types import Dataset, ReaderOptions


logger = logging.getLogger(__name__)


def get_table_row_count(parquet_url: str):
    spark = get_spark_session()
    return spark.read.parquet(parquet_url).count()


def collate_and_preprocess(batch_preprocessor: BatchPreprocessor, use_gpu: bool):
    """ Helper for Petastorm's DataLoader to preprocess.
    TODO(kaiwenw): parallelize preprocessing by using transform of Petastorm reader
    Should pin memory and preprocess in reader and convert to gpu in collate_fn.
    """

    def collate_fn(batch_list: List[Dict]):
        batch = decimal_friendly_collate(batch_list)
        preprocessed_batch = batch_preprocessor(batch)
        if use_gpu:
            preprocessed_batch = preprocessed_batch.cuda()
        return preprocessed_batch

    return collate_fn


def get_petastorm_dataloader(
    dataset: Dataset,
    batch_size: int,
    batch_preprocessor: BatchPreprocessor,
    use_gpu: bool,
    reader_options: ReaderOptions,
):
    """ get petastorm loader for dataset (with preprocessor) """
    data_reader = make_batch_reader(
        dataset.parquet_url,
        num_epochs=1,
        reader_pool_type=reader_options.petastorm_reader_pool_type,
    )
    # NOTE: must be wrapped by DataLoaderWrapper to call __exit__() on end of epoch
    return DataLoader(
        data_reader,
        batch_size=batch_size,
        collate_fn=collate_and_preprocess(
            batch_preprocessor=batch_preprocessor, use_gpu=use_gpu
        ),
    )


def gather_eval_data(
    trainer: RLTrainer,
    eval_dataset: Dataset,
    batch_preprocessor: BatchPreprocessor,
    use_gpu: bool,
    reader_options: ReaderOptions,
) -> EvaluationDataPage:
    """ Sorts, computes logged values and validates the EvaluationDataPage """
    if isinstance(trainer, (SACTrainer, TD3Trainer)):
        raise NotImplementedError("TODO: Implement CPE for continuous algos")
    assert (
        trainer.calc_cpe_in_training
    ), "this function should only be called when this is true."

    # first read the eval_dataset as EvaluationDataPages
    device = "cuda" if use_gpu else "cpu"
    eval_data = None
    with make_batch_reader(
        eval_dataset.parquet_url,
        num_epochs=1,
        reader_pool_type=reader_options.petastorm_reader_pool_type,
    ) as reader:
        for batch in reader:
            assert rlt.isinstance_namedtuple(batch)
            tensor_batch = dict_to_tensor(batch._asdict(), device=device)
            tdp: rlt.PreprocessedTrainingBatch = batch_preprocessor(tensor_batch)
            edp = EvaluationDataPage.create_from_training_batch(tdp, trainer)
            if eval_data is None:
                eval_data = edp
            else:
                eval_data = eval_data.append(edp)

    eval_data = eval_data.sort()
    eval_data = eval_data.compute_values(trainer.gamma)
    eval_data.validate()
    return eval_data


def train_and_evaluate_generic(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    trainer: RLTrainer,
    num_epochs: int,
    use_gpu: bool,
    batch_preprocessor: BatchPreprocessor,
    reporter: Observer,
    evaluator: Evaluator,
    reader_options: Optional[ReaderOptions] = None,
) -> None:
    reader_options = reader_options or ReaderOptions()
    epoch_iterator = EpochIterator(num_epochs=num_epochs)
    train_dataset_size = get_table_row_count(train_dataset.parquet_url)
    # pyre-fixme[16]: `EpochIterator` has no attribute `add_observer`.
    for epoch in epoch_iterator.add_observer(reporter):
        logger.info(f"Starting training epoch {epoch}.")
        dataloader = get_petastorm_dataloader(
            dataset=train_dataset,
            # pyre-fixme[6]: Expected `int` for 2nd param but got `Optional[int]`.
            batch_size=trainer.minibatch_size,
            batch_preprocessor=batch_preprocessor,
            use_gpu=use_gpu,
            reader_options=reader_options,
        )
        dataloader_wrapper = DataLoaderWrapper(
            dataloader=dataloader, dataloader_size=train_dataset_size
        )
        for batch in dataloader_wrapper:
            trainer.train(batch)

        if eval_dataset is not None:
            eval_data = gather_eval_data(
                trainer=trainer,
                eval_dataset=eval_dataset,
                batch_preprocessor=batch_preprocessor,
                use_gpu=use_gpu,
                reader_options=reader_options,
            )
            # evaluator passes cpe_details to reporter via notify_observers
            evaluator.evaluate_post_training(eval_data)


# TODO: Move this to appropriate location
class PetastormLightningDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, batch_preprocessor, reader_options):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_preprocessor = batch_preprocessor
        self.reader_options = reader_options

    def _closing_iter(self, dataloader):
        yield from dataloader
        dataloader.__exit__(None, None, None)

    def train_dataloader(self):
        dataloader = get_petastorm_dataloader(
            dataset=self.train_dataset,
            batch_size=self.reader_options.minibatch_size,
            batch_preprocessor=self.batch_preprocessor,
            use_gpu=False,
            reader_options=self.reader_options,
        )
        return self._closing_iter(dataloader)

    def test_dataloader(self):
        dataloader = get_petastorm_dataloader(
            dataset=self.eval_dataset,
            batch_size=self.reader_options.minibatch_size,
            batch_preprocessor=self.batch_preprocessor,
            use_gpu=False,
            reader_options=self.reader_options,
        )
        return self._closing_iter(dataloader)


def train_eval_lightning(
    train_dataset,
    eval_dataset,
    trainer_module,
    num_epochs,
    use_gpu,
    batch_preprocessor=None,
    reader_options: Optional[ReaderOptions] = None,
    checkpoint_path: Optional[str] = None,
) -> pl.Trainer:
    reader_options = reader_options or ReaderOptions()
    datamodule = PetastormLightningDataModule(
        train_dataset, eval_dataset, batch_preprocessor, reader_options
    )
    # pyre-fixme[16]: Module `pl` has no attribute `Trainer`.
    # pyre-fixme[16]: Module `pl` has no attribute `Trainer`.
    trainer = pl.Trainer(
        max_epochs=num_epochs * 1000,
        gpus=int(use_gpu),
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=checkpoint_path,
        callbacks=[StoppingEpochCallback(num_epochs)],
    )
    trainer.fit(trainer_module, datamodule=datamodule)
    # TODO: evaluate
    return trainer
