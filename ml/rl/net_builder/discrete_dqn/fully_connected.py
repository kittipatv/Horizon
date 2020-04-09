#!/usr/bin/env python3

from typing import Dict, List, Type

from ml.rl import types as rlt
from ml.rl.core.dataclasses import dataclass, field
from ml.rl.models.base import ModelBase
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.net_builder.discrete_dqn_net_builder import DiscreteDQNNetBuilder
from ml.rl.parameters import NormalizationParameters, param_hash


@dataclass
class FullyConnected(DiscreteDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    dropout_ratio: float = 0.0

    def __post_init_post_parse__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_feature_config: rlt.ModelFeatureConfig,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        output_dim: int,
    ) -> ModelBase:
        state_dim = self._get_input_dim(state_normalization_parameters)
        return FullyConnectedDQN(
            state_dim=state_dim,
            action_dim=output_dim,
            sizes=self.sizes,
            activations=self.activations,
            dropout_ratio=self.dropout_ratio,
        )
