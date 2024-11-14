# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_extended_tp:
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
            )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize experts
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules.experts, MLPSubmodules):
                self.experts = TEGroupedMLP(
                    self.num_local_experts, self.config, self.submodules.experts
                )
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules.experts, MLPSubmodules)
            self.experts = SequentialMLP(
                self.num_local_experts, self.config, self.submodules.experts
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(self.config, self.submodules.shared_experts)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output += self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias


# from functools import partial
# from megatron.core.transformer.moe import megablocks_utils

# class dMoELayer(BaseMoELayer):
#     def __init__(
#         self, 
#         config: TransformerConfig, 
#         layer_number: int = None
#     ):
#         megablocks_utils.assert_megablocks_is_available()
#         super(dMoELayer, self).__init__(config=config, layer_number=layer_number)
#         # assert self.config.moe_layer_recompute, 'IDGAF'
#         # assert self.config.moe_token_dispatcher_type == "allgather", 'IDGAF'
#         # assert not self.config.moe_shared_expert_intermediate_size
#         # assert not self.config.moe_shared_expert_overlap

#         args = {
#             'hidden_size': self.config.hidden_size,
#             'ffn_hidden_size': self.config.ffn_hidden_size,
#             'moe_top_k': self.config.moe_router_topk,
#             'activation_fn': self.config.activation_func,
#             'moe_jitter_eps': self.config.moe_input_jitter_eps,  # Disable randomiztion
#             'moe_normalize_expert_weights': 1.0,
#             'uniform_expert_assignment': False,
#             'bias': False,
#             'device': torch.cuda.current_device(),
#             'moe_num_experts': self.config.num_moe_experts,
#             'mlp_type': 'glu',
#             'fp16': config.fp16,
#             'bf16': config.bf16,
#             'init_method': partial(torch.nn.init.normal_, mean=0.0, std=0.02),
#         }
#         if self.expert_parallel_size > 1:
#             ep_args = {
#                 'moe_expert_model_parallelism': True,
#                 'expert_parallel_group': parallel_state.get_expert_model_parallel_group(),
#             }
#             args.update(ep_args)
#         args = megablocks_utils.arguments(**args,)
#         self.dmoe = megablocks_utils.dmoe(args)
#         self.router = self.dmoe.router

#     def forward(self, hidden_states: torch.Tensor):
#         return self.dmoe.forward(hidden_states)
    
#     def set_layer_number(self, layer_number: int):
#         return