#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
The Trainer for Cross-Entropy Method. The idea is that an ensemble of
 world models are fitted to predict transitions and reward functions.
A cross entropy method-based planner will then plan the best next action
based on simulation data generated by the fitted world models.

The idea is inspired by: https://arxiv.org/abs/1805.12114
"""
import logging
from typing import List

import reagent.types as rlt
from reagent.models.cem_planner import CEMPlannerNetwork
from reagent.parameters import CEMTrainerParameters
from reagent.training.rl_trainer_pytorch import RLTrainer
from reagent.training.world_model.mdnrnn_trainer import MDNRNNTrainer


logger = logging.getLogger(__name__)


def print_mdnrnn_losses(minibatch, model_index, losses) -> None:
    logger.info(
        f"{minibatch}-th minibatch {model_index}-th model: \n"
        f'loss={losses["loss"]}, bce={losses["bce"]}, '
        f'gmm={losses["gmm"]}, mse={losses["mse"]}\n'
    )


class CEMTrainer(RLTrainer):
    def __init__(
        self,
        cem_planner_network: CEMPlannerNetwork,
        world_model_trainers: List[MDNRNNTrainer],
        parameters: CEMTrainerParameters,
        use_gpu: bool = False,
    ) -> None:
        super().__init__(parameters.rl, use_gpu=use_gpu)
        self.cem_planner_network = cem_planner_network
        self.world_model_trainers = world_model_trainers
        self.minibatch_size = parameters.mdnrnn.minibatch_size

    def train(self, training_batch: rlt.MemoryNetworkInput) -> None:
        for i, trainer in enumerate(self.world_model_trainers):
            losses = trainer.train(training_batch)
            # TODO: report losses instead of printing them
            # print_mdnrnn_losses(self.minibatch, i, losses)

        self.minibatch += 1
