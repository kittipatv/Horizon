#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
This file shows an example of using embedded states to feed to RL models in
partially observable environments (POMDPs). Embedded states are generated by a world
model which learns how to encode past n observations into a low-dimension
vector.Embedded states improve performance in POMDPs compared to just using
one-step observations as states because they encode more historical information
than one-step observations.
"""
import argparse
import json
import logging
import sys
from collections import deque

import gym
import numpy as np
import reagent.types as rlt
import torch
from gym import Env
from gym.spaces import Box
from reagent.json_serialize import from_json, json_to_object
from reagent.models.world_model import MemoryNetwork
from reagent.test.gym.open_ai_gym_environment import EnvType, OpenAIGymEnvironment
from reagent.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool
from reagent.test.gym.run_gym import (
    OpenAiGymParameters,
    create_epsilon,
    create_predictor,
    create_trainer,
    train_gym_offline_rl,
)
from reagent.test.gym.world_model.mdnrnn_gym import create_embed_rl_dataset, mdnrnn_gym
from reagent.training.rl_dataset import RLDataset


logger = logging.getLogger(__name__)


class StateEmbedGymEnvironment(Env):
    def __init__(
        self,
        gym_env: Env,
        mdnrnn: MemoryNetwork,
        max_embed_seq_len: int,
        state_min_value: float,
        state_max_value: float,
    ):
        self.env = gym_env
        self.unwrapped.spec = self.env.unwrapped.spec
        self.max_embed_seq_len = max_embed_seq_len
        self.mdnrnn = mdnrnn
        self.embed_size = self.mdnrnn.num_hiddens
        self.raw_state_dim = self.env.observation_space.shape[0]  # type: ignore
        self.state_dim = self.embed_size + self.raw_state_dim
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_type = EnvType.DISCRETE_ACTION
            self.action_dim = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.action_type = EnvType.CONTINUOUS_ACTION
            self.action_dim = self.env.action_space.shape[0]

        self.action_space = self.env.action_space
        self.observation_space = Box(  # type: ignore
            low=state_min_value, high=state_max_value, shape=(self.state_dim,)
        )

        self.cur_raw_state = None
        self.recent_states = deque([], maxlen=self.max_embed_seq_len)  # type: ignore
        self.recent_actions = deque([], maxlen=self.max_embed_seq_len)  # type: ignore

    def seed(self, seed):
        self.env.seed(seed)

    def embed_state(self, state):
        """ Embed state after either reset() or step() """
        assert len(self.recent_states) == len(self.recent_actions)
        old_mdnrnn_mode = self.mdnrnn.mdnrnn.training
        self.mdnrnn.mdnrnn.eval()

        # Embed the state as the hidden layer's output
        # until the previous step + current state
        if len(self.recent_states) == 0:
            mdnrnn_state = np.zeros((1, self.raw_state_dim))
            mdnrnn_action = np.zeros((1, self.action_dim))
        else:
            mdnrnn_state = np.array(list(self.recent_states))
            mdnrnn_action = np.array(list(self.recent_actions))

        mdnrnn_state = torch.tensor(mdnrnn_state, dtype=torch.float).unsqueeze(1)
        mdnrnn_action = torch.tensor(mdnrnn_action, dtype=torch.float).unsqueeze(1)
        mdnrnn_input = rlt.PreprocessedStateAction.from_tensors(
            state=mdnrnn_state, action=mdnrnn_action
        )
        mdnrnn_output = self.mdnrnn(mdnrnn_input)
        hidden_embed = (
            mdnrnn_output.all_steps_lstm_hidden[-1].squeeze().detach().cpu().numpy()
        )
        state_embed = np.hstack((hidden_embed, state))
        self.mdnrnn.mdnrnn.train(old_mdnrnn_mode)
        logger.debug(
            "Embed_state\nrecent states: {}\nrecent actions: {}\nstate_embed{}\n".format(
                np.array(self.recent_states), np.array(self.recent_actions), state_embed
            )
        )
        return state_embed

    def reset(self):
        next_raw_state = self.env.reset()
        self.recent_states = deque([], maxlen=self.max_embed_seq_len)
        self.recent_actions = deque([], maxlen=self.max_embed_seq_len)
        self.cur_raw_state = next_raw_state
        next_embed_state = self.embed_state(next_raw_state)
        return next_embed_state

    def step(self, action):
        if self.action_type == EnvType.DISCRETE_ACTION:
            action_np = np.zeros(self.action_dim)
            action_np[action] = 1.0
        else:
            action_np = action
        self.recent_states.append(self.cur_raw_state)
        self.recent_actions.append(action_np)
        next_raw_state, reward, terminal, info = self.env.step(action)
        logger.debug("action {}, reward {}\n".format(action, reward))
        self.cur_raw_state = next_raw_state
        next_embed_state = self.embed_state(next_raw_state)
        return next_embed_state, reward, terminal, info


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in an OpenAI Gym environment. "
        "States are embedded by a mdn-rnn model."
    )
    parser.add_argument(
        "-p",
        "--mdnrnn_parameters",
        help="Path to JSON parameters file for MDN-RNN training.",
    )
    parser.add_argument(
        "-q", "--rl_parameters", help="Path to JSON parameters file for RL training."
    )
    parser.add_argument(
        "-s",
        "--score-bar",
        help="Bar for averaged tests scores.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="If set, will use GPU with specified ID. Otherwise will use CPU.",
        default=-1,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="If set, use logging level specified (debug, info, warning, error, "
        "critical). Else defaults to info.",
        default="info",
    )
    args = parser.parse_args(args)
    if args.log_level not in ("debug", "info", "warning", "error", "critical"):
        raise Exception("Logging level {} not valid level.".format(args.log_level))
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    with open(args.mdnrnn_parameters, "r") as f:
        mdnrnn_params = json_to_object(f.read(), OpenAiGymParameters)
    with open(args.rl_parameters, "r") as f:
        rl_params = json_to_object(f.read(), OpenAiGymParameters)

    env, mdnrnn_trainer, embed_rl_dataset = create_mdnrnn_trainer_and_embed_dataset(
        mdnrnn_params, rl_params.use_gpu
    )

    max_embed_seq_len = mdnrnn_params["run_details"]["seq_len"]
    _, _, rl_trainer, rl_predictor, state_embed_env = run_gym(
        rl_params,
        args.score_bar,
        embed_rl_dataset,
        env.env,
        mdnrnn_trainer.mdnrnn,
        max_embed_seq_len,
    )


def create_mdnrnn_trainer_and_embed_dataset(
    mdnrnn_params: OpenAiGymParameters, use_gpu
):
    env, mdnrnn_trainer, _, _, _ = mdnrnn_gym(mdnrnn_params)
    embed_rl_dataset = RLDataset("/tmp/rl.pkl")
    create_embed_rl_dataset(
        env, mdnrnn_trainer, embed_rl_dataset, use_gpu, mdnrnn_params.run_details
    )
    return env, mdnrnn_trainer, embed_rl_dataset


def run_gym(
    params: OpenAiGymParameters,
    score_bar,
    embed_rl_dataset: RLDataset,
    gym_env: Env,
    mdnrnn: MemoryNetwork,
    max_embed_seq_len: int,
):
    assert params.rl is not None
    rl_parameters = params.rl

    env_type = params.env
    model_type = params.model_type
    epsilon, epsilon_decay, minimum_epsilon = create_epsilon(
        offline_train=True, rl_parameters=rl_parameters, params=params
    )

    replay_buffer = OpenAIGymMemoryPool(params.max_replay_memory_size)
    for row in embed_rl_dataset.rows:
        replay_buffer.insert_into_memory(**row)

    assert replay_buffer.memory_buffer is not None
    state_mem = replay_buffer.memory_buffer.state
    state_min_value = torch.min(state_mem).item()
    state_max_value = torch.max(state_mem).item()
    state_embed_env = StateEmbedGymEnvironment(
        gym_env, mdnrnn, max_embed_seq_len, state_min_value, state_max_value
    )
    open_ai_env = OpenAIGymEnvironment(
        state_embed_env,
        epsilon,
        rl_parameters.softmax_policy,
        rl_parameters.gamma,
        epsilon_decay,
        minimum_epsilon,
    )
    rl_trainer = create_trainer(params, open_ai_env)
    rl_predictor = create_predictor(
        rl_trainer, model_type, params.use_gpu, open_ai_env.action_dim
    )

    assert (
        params.run_details.max_steps is not None
        and params.run_details.offline_train_epochs is not None
    ), "Missing data required for offline training: {}".format(str(params.run_details))
    return train_gym_offline_rl(
        gym_env=open_ai_env,
        replay_buffer=replay_buffer,
        model_type=model_type,
        trainer=rl_trainer,
        predictor=rl_predictor,
        test_run_name="{} offline rl state embed".format(env_type),
        score_bar=score_bar,
        max_steps=params.run_details.max_steps,
        avg_over_num_episodes=params.run_details.avg_over_num_episodes,
        offline_train_epochs=params.run_details.offline_train_epochs,
        num_batch_per_epoch=None,
    )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    args = sys.argv
    main(args[1:])
