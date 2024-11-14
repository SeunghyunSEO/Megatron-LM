'''
https://github.com/databricks/megablocks/blob/main/tests/layers/dmoe_test.py
https://github.com/mosaicml/llm-foundry/blob/main/tests/models/layers/test_dmoe.py

## from llm foundry
https://github.com/mosaicml/llm-foundry/blob/ed6b72bfa555484e41a7e48af132af5692acefd3/llmfoundry/models/layers/ffn.py#L533
https://github.com/mosaicml/llm-foundry/blob/ed6b72bfa555484e41a7e48af132af5692acefd3/llmfoundry/models/layers/ffn.py#L470

## from olmo
https://github.com/allenai/OLMo/blob/sewon-olmoe/olmo/model.py#L678-L690
https://github.com/allenai/OLMo/blob/8589c38756ac3359abbe5938dfbeaff20e92c3a1/olmo/model.py#L1460
https://github.com/allenai/OLMo/blob/sewon-olmoe/olmo/config.py#L1340
https://github.com/databricks/megablocks/blob/84286de8ab5be0c73928a0059f50c7e2b650e4b1/megablocks/layers/arguments.py#L23
https://github.com/databricks/megablocks/issues/57
'''

import os
import copy
from functools import partial
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Shard, Replicate
from torch.distributed import ProcessGroup

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from dmoe.dmoe import dMoE
from utils import (
    set_seed,
    init_dist,
    print_message_with_master_process,
    _get_torch_dtype,
    cleanup_distributed,
)

try:
    import megablocks
    is_megablocks_imported = True
except ModuleNotFoundError:
    is_megablocks_imported = False

def _get_all_inputs(
    input_shape: list[int],
    dtype: Optional[torch.dtype],
):
    world_size: int = dist.get_world_size()
    rank: int = dist.get_rank()
    device: torch.device = torch.device(f'cuda:{rank}')
    all_inputs = [
        torch.rand(
            input_shape,
            device=device,
            dtype=dtype,
        ) for _ in range(world_size)
    ]
    return all_inputs

def test_dmoe(
    moe_num_experts: int = 8,
    mlp_type: str = 'glu',
    moe_normalize_expert_weights: Union[float, int] = 1.0,
    two_d_input: bool = False,
    use_fsdp: bool = False,
    use_only_fsdp: bool = False,
):
    ## init
    assert is_megablocks_imported, "you should install megablocks for comparison"
    rank, world_rank, world_size, device = init_dist()
    # rank = dist.get_rank()
    # device = torch.device(f'cuda:{dist.get_rank()}')
    DP, SHARD, EP = int(os.environ["DP"]), int(os.environ["SHARD"]), int(os.environ["EP"])
    assert world_size == (DP*SHARD*EP)
    assert SHARD == 1, "currently HSDP is not supported"
    assert (moe_num_experts >= EP) or (moe_num_experts % EP) == 0, 'Mismatch between EP and moe_num_experts.'
    set_seed()

    ## moe configs
    moe_top_k = min(2, moe_num_experts)

    # Generate inputs
    batch_size = 2
    seq_len = 3
    hidden_size = 256
    if two_d_input:
        input_shape = [batch_size * seq_len, hidden_size]
    else:
        input_shape = [batch_size, seq_len, hidden_size]

    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    param_dtype, reduce_dtype = torch.bfloat16, torch.float32
    fp16, bf16 = False, True
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    dtype = _get_torch_dtype(fp16, bf16)
    x = _get_all_inputs(input_shape, dtype)[rank]

    ## Construct torch fsdp dMoE
    common_args = {
        'hidden_size': hidden_size,
        'ffn_hidden_size': hidden_size,
        'moe_top_k': moe_top_k,
        'activation_fn': partial(F.gelu, approximate='none'),
        'moe_jitter_eps': 0.0,  # Disable randomiztion
        'moe_normalize_expert_weights': moe_normalize_expert_weights,
        'uniform_expert_assignment': False,
        'bias': False,
        'device': device,
        'moe_num_experts': moe_num_experts,
        'mlp_type': mlp_type,
    }
    torch_dmoe = dMoE(**common_args).to(device, dtype=dtype)
    if use_fsdp:
        torch_dmoe_mesh = init_device_mesh(
            "cuda", 
            (world_size,), 
            mesh_dim_names=("fsdp",),
        )
        # ## FSDP 1
        # torch_dmoe = FSDP(
        #     torch_dmoe, 
        #     device_mesh=torch_dmoe_mesh["fsdp"], 
        # )
        ## FSDP 2
        fsdp_config = {"mesh": torch_dmoe_mesh["fsdp"], "mp_policy": mp_policy}
        fully_shard(torch_dmoe, **fsdp_config, reshard_after_forward=True)
        
    else:
        torch_dmoe_mesh = None
        torch_dmoe = DDP(
            torch_dmoe,
            device_ids=[rank],
        )
    torch_dmoe_optimizer = optim.SGD(torch_dmoe.parameters(), lr=0.1)

    ## Construct megablocks ep+fsdp dMoE
    mp_dmoe_args = copy.deepcopy(common_args)
    extra_args = {
        'fp16': fp16,
        'bf16': bf16,
        'init_method': partial(torch.nn.init.uniform_, a=-1.0, b=1.0),
    }
    mb_dmoe_mesh = None
    if EP > 1:
        if use_fsdp and use_only_fsdp:
            # test only FSDP first
            mb_dmoe_mesh = init_device_mesh(
                'cuda',
                (world_size,),
                mesh_dim_names=('fsdp',),
            )
            DP_mesh = mb_dmoe_mesh['fsdp']
            EP_mesh = None
        else:
            mb_dmoe_mesh = init_device_mesh(
                'cuda',
                (DP*SHARD, EP),
                mesh_dim_names=('weight_parallel', 'expert_parallel'),
            )
            DP_mesh = mb_dmoe_mesh['weight_parallel']
            EP_mesh = mb_dmoe_mesh['expert_parallel']
            extra_args.update(
                {
                    'moe_expert_model_parallelism': True,
                    'expert_parallel_group': EP_mesh.get_group(0), # https://github.com/databricks/megablocks/blob/7b0337fa7278d224bf0c9be71c3a92c392fdd340/megablocks/layers/moe.py#L278
                },
            )
    mp_dmoe_args.update(extra_args)
    args = megablocks.layers.arguments.Arguments(**mp_dmoe_args)
    mb_dmoe = megablocks.layers.dmoe.dMoE(args).to(device)

    ## Construct a DTensor from an already sharded local parameter.
    ## https://pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.DTensor.from_local
    def dtensorify_param(
        param: nn.Parameter,
        mesh: DeviceMesh,
        placements: list[Placement],
    ):
        param_dtensor = DTensor.from_local(
            param.data,
            device_mesh=mesh,
            placements=placements,
            run_check=False,
        )
        return nn.Parameter(param_dtensor)

    if use_fsdp:

        '''
        ## should i use custom policy for EP + FSDP? (FSDP for other layers like router, qkvo)  
        https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel
        https://github.com/mosaicml/llm-foundry/blob/ed6b72bfa555484e41a7e48af132af5692acefd3/llmfoundry/models/layers/ffn.py#L421
        https://github.com/mosaicml/composer/blob/f542250ca3c35977e5f1a153b78177d675de53dd/composer/distributed/dist_strategy.py#L460
        
        it's like ...
        def custom_auto_wrap_policy(
            module: nn.Module,
            recurse: bool,
            nonwrapped_numel: int,
            # Additional custom arguments
            min_num_params: int = int(1e8),
        ) -> bool:
            return nonwrapped_numel >= min_num_params
        # Configure a custom `min_num_params`
        my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))

        ## dmoe does not support HSDP ?
        https://github.com/mosaicml/llm-foundry/blob/ed6b72bfa555484e41a7e48af132af5692acefd3/llmfoundry/models/layers/ffn.py#L414

        ## how bound FSDP 2?
        NotImplementedError: aten._foreach_add_.List: DTensor does not support cross-mesh operation yet! 
        Got meshes: DeviceMesh('cuda', [0], mesh_dim_names=('weight_parallel',)) DeviceMesh('cuda', [0, 1], mesh_dim_names=('expert_parallel',))
        '''
        
        fsdp_config = {"mesh": DP_mesh, "mp_policy": mp_policy}
        if use_only_fsdp:
            fully_shard(mb_dmoe, **fsdp_config, reshard_after_forward=False)
        else:
            expert_placements: list[Placement] = [Shard(0)]
            # Register in two loops as you cannot overwrite parameters while iterating over named_parameters()
            dtensorified_params = [(
                name,
                dtensorify_param(
                    param=parameter,
                    mesh=EP_mesh,
                    placements=expert_placements,
                ),
            ) for name, parameter in mb_dmoe.experts.mlp.named_parameters()]
            for name, dtensorified_param in dtensorified_params:
                mb_dmoe.experts.mlp.register_parameter(name, dtensorified_param)
            fully_shard(mb_dmoe.router, **fsdp_config, reshard_after_forward=False)
        mb_dmoe_optimizer = optim.SGD(mb_dmoe.parameters(), lr=0.1)

        print_message_with_master_process(rank, f'''
        **********************************************
        ***** before copying params from mb dmoe *****
        **********************************************
        ''')
        for (n, p), (n_, p_) in zip(torch_dmoe.named_parameters(), mb_dmoe.named_parameters()):
            print_message_with_master_process(rank, f'torch_dmoe/{n}: {p}')
            print_message_with_master_process(rank, f'mb_dmoe/{n_}: {p_}')

        '''
        [rank0]:        While copying the parameter named "experts.mlp.w2", whose dimensions in the model are torch.Size([2048, 256]) 
        and whose dimensions in the checkpoint are torch.Size([2048, 256]), 
        an exception occurred : ("aten.copy_.default: DTensor does not support cross-mesh operation yet! 
        Got meshes: DeviceMesh('cuda', [0, 1], mesh_dim_names=('fsdp',)) DeviceMesh('cuda', [[0, 1]], mesh_dim_names=('weight_parallel', 'expert_parallel'))",).
        '''
        mb_dmoe_state_dict = get_model_state_dict(
            mb_dmoe,
            options=StateDictOptions(full_state_dict=True,),
        )
        for k, v in mb_dmoe_state_dict.items():
            mb_dmoe_state_dict[k] = DTensor.from_local(
                v.data,
                device_mesh=torch_dmoe_mesh,
            )
        torch_dmoe.load_state_dict(mb_dmoe_state_dict, strict=True)
        print_message_with_master_process(rank, f'''
        *********************************************
        ***** after copying params from mb dmoe *****
        *********************************************
        ''')
        for (n, p), (n_, p_) in zip(torch_dmoe.named_parameters(), mb_dmoe.named_parameters()):
            print_message_with_master_process(rank, f'torch_dmoe/{n}: {p}')
            print_message_with_master_process(rank, f'mb_dmoe/{n_}: {p_}')

    else:
        mb_dmoe.router = DDP(
            mb_dmoe.router, 
            device_ids=[rank]
        )

        if EP > 1:
            assert mb_dmoe_mesh is not None
            two_d_placements: list[Placement] = [Replicate(), Shard(0)]
            dtensorified_params = [(
                name,
                dtensorify_param(
                    param=parameter,
                    mesh=mb_dmoe_mesh,
                    placements=two_d_placements,
                ),
            ) for name, parameter in mb_dmoe.experts.mlp.named_parameters()]
            tp_names = []
            for name, dtensorified_param in dtensorified_params:
                mb_dmoe.experts.mlp.register_parameter(name, dtensorified_param)
                tp_names.append('experts.mlp.' + name)
            _pre_dp_module_transform(mb_dmoe.experts.mlp)

            mb_dmoe.experts = DDP(
                mb_dmoe.experts, 
                process_group=DP_mesh.get_group(0)
            )

            # Copy mb_dmoe's parameters to torch_dmoe
            mb_dmoe_state_dict = get_model_state_dict(
                mb_dmoe,
                options=StateDictOptions(full_state_dict=True,),
            )
            for key, t in mb_dmoe_state_dict.items():
                if key in tp_names:
                    dtensor_full = DTensor.from_local(
                        t,  # pyright: ignore[reportGeneralTypeIssues]
                        device_mesh=mb_dmoe_mesh,
                        placements=two_d_placements,
                    ).full_tensor()
                    mb_dmoe_state_dict[key] = dtensor_full
        else:
            mb_dmoe.experts = DDP(
                mb_dmoe.experts, 
                device_ids=[rank]
            )

            # Copy mb_dmoe's parameters to torch_dmoe
            mb_dmoe_state_dict = get_model_state_dict(
                mb_dmoe,
                options=StateDictOptions(full_state_dict=True,),
            )
        mb_dmoe_optimizer = optim.SGD(mb_dmoe.parameters(), lr=0.1)

        # Load mb_dmoe state dict to torch dmoe
        torch_dmoe.module.load_state_dict(mb_dmoe_state_dict, strict=True)
        
    ## sanity check
    print_message_with_master_process(rank, f'''
    torch_dmoe_mesh: {torch_dmoe_mesh}
    mb_dmoe_mesh: {mb_dmoe_mesh}
    DP_mesh: {DP_mesh}
    EP_mesh: {EP_mesh}

    common_args: {common_args}
    mp_dmoe_args: {mp_dmoe_args}
    args: {args}

    torch_dmoe: {torch_dmoe}
    mb_dmoe: {mb_dmoe}
    ''')

    # Run train_step check
    torch_y = torch_dmoe(x)
    print(f'''
    rank: {rank}
    x.size(): {x.size()}
    x.device: {x.device}
    torch_y.xise(): {torch_y.size()}
    ''')
    mb_y = mb_dmoe(x)

    torch_y.sum().backward()
    mb_y.sum().backward()
    torch_dmoe_optimizer.step()
    mb_dmoe_optimizer.step()

    torch_y = torch_dmoe(x)
    mb_y = mb_dmoe(x)
    torch.testing.assert_close(torch_y, mb_y)
    cleanup_distributed()

if __name__ == "__main__":
    use_fsdp=False
    use_fsdp=True
    use_only_fsdp=False
    # use_only_fsdp=True
    test_dmoe(use_fsdp=use_fsdp, use_only_fsdp=use_only_fsdp)