#!/usr/bin/env python3

import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from reagent.models.dqn import FullyConnectedDQN
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example(
    #rank, world_size
):
    # create default process group
    dist.init_process_group("gloo")
    # create local model
    rank = dist.get_rank()

    logger.info(f"my rank is {rank}")

    torch.manual_seed(101)
    trainer_group = dist.new_group([i * 2 for i in range(2)])

    if rank % 2 == 0:
        model = FullyConnectedDQN(5, 4, sizes=[16], activations=["relu"])
        ddp = DDP(model, process_group=trainer_group)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(ddp.parameters())

        torch.manual_seed(rank)

        input = model.input_prototype()

        labels = torch.randn(1, 4)

        for i in range(100):
            optimizer.zero_grad()
            output = ddp(input)
            loss = loss_fn(output.q_values, labels)
            if rank == 0:
                logger.info(f"Loss at iteration {i}: {loss.detach().item()}")
            loss.backward()
            optimizer.step()

    else:
        logger.info(f"I'm not a trainer")

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    example()
