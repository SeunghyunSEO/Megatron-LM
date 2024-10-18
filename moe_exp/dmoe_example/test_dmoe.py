# https://github.com/mosaicml/llm-foundry/blob/main/tests/models/layers/test_dmoe.py

import os
import copy
from contextlib import nullcontext
from functools import partial
from typing import Optional, Union

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed._tensor import DTensor, Placement, Replicate, Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dmoe import dMoE
from src.ffn import dtensorify_param

try:
    import megablocks
    is_megablocks_imported = True
except ModuleNotFoundError:
    is_megablocks_imported = False

from pdb import set_trace as Tra

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_dist():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    print(f"rank: {rank}, world size: {world_size}")
    return rank, world_size

def print_message_with_master_process(rank, message):
    if rank==0:
        print(message)

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

def _get_torch_dtype(fp16: bool, bf16: bool) -> Optional[torch.dtype]:
    if fp16:
        return torch.float16
    elif bf16:
        return torch.bfloat16
    return None

def test_dmoe(
    moe_num_experts: int = 8,
    mlp_type: str = 'glu',
    moe_world_size: int = 2,
    moe_normalize_expert_weights: Union[float, int] = 1.0,
    two_d_input: bool = False,
):
    assert is_megablocks_imported, "you should install megablocks for comparison"
    init_dist()

    # moe configs
    if moe_world_size > moe_num_experts or moe_num_experts % moe_world_size != 0:
        pytest.skip('Mismatch between moe_world_size and moe_num_experts.')
    moe_top_k = min(2, moe_num_experts)

    # Generate inputs
    rank = dist.get_rank()
    batch_size = 2
    seq_len = 3
    hidden_size = 256
    if two_d_input:
        input_shape = [batch_size * seq_len, hidden_size]
    else:
        input_shape = [batch_size, seq_len, hidden_size]
    fp16 = False
    bf16 = True
    dtype = _get_torch_dtype(fp16, bf16)
    x = _get_all_inputs(input_shape, dtype)[rank]

    # Construct DDP torch dMoE
    device = torch.device(f'cuda:{dist.get_rank()}')
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
    torch_dmoe = DDP(
        torch_dmoe,
        device_ids=[rank],
    )
    torch_dmoe_optimizer = optim.SGD(torch_dmoe.parameters(), lr=0.1)

    # Construct TP MB dMoE
    mp_dmoe_args = copy.deepcopy(common_args)
    extra_args = {
        'fp16': fp16,
        'bf16': bf16,
        'init_method': partial(torch.nn.init.uniform_, a=-1.0, b=1.0),
    }

    # https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
    device_mesh = None
    if moe_world_size > 1:
        world_size = dist.get_world_size()
        assert world_size % moe_world_size == 0
        moe_dp_dim = world_size // moe_world_size
        device_mesh = init_device_mesh(
            'cuda',
            (moe_dp_dim, moe_world_size),
            mesh_dim_names=('weight_parallel', 'expert_parallel'),
        )
        expert_parallel_group = device_mesh['expert_parallel'].get_group(0)
        extra_args.update({
            'moe_expert_model_parallelism': True,
            'expert_parallel_group': expert_parallel_group, # https://github.com/databricks/megablocks/blob/7b0337fa7278d224bf0c9be71c3a92c392fdd340/megablocks/layers/moe.py#L278
        },)
    mp_dmoe_args.update(extra_args)
    args = megablocks.layers.arguments.Arguments(**mp_dmoe_args,)
    mb_dmoe = megablocks.layers.dmoe.dMoE(args).to(device)
    mb_dmoe.router = DDP(
        mb_dmoe.router, 
        device_ids=[rank]
    )

    ## sanity check
    print_message_with_master_process(rank, f'''
    common_args: {common_args}
    torch_dmoe: {torch_dmoe}

    world_size: {world_size}
    moe_world_size: {moe_world_size}
    moe_dp_dim: {moe_dp_dim}
    device_mesh: {device_mesh}
    expert_parallel_group: {expert_parallel_group}
    expert_parallel_group.size(): {expert_parallel_group.size()}
    expert_parallel_group.rank(): {expert_parallel_group.rank()}

    mp_dmoe_args: {mp_dmoe_args}
    mb_dmoe: {mb_dmoe}
    mb_dmoe.router: {mb_dmoe.router}
    ''')

    if moe_world_size > 1:
        assert device_mesh is not None
        two_d_placements: list[Placement] = [Replicate(), Shard(0)]
        dtensorified_params = [(
            name,
            dtensorify_param(
                param=parameter,
                mesh=device_mesh,
                placements=two_d_placements,
            ),
        ) for name, parameter in mb_dmoe.experts.mlp.named_parameters()]
        tp_names = []
        for name, dtensorified_param in dtensorified_params:
            mb_dmoe.experts.mlp.register_parameter(name, dtensorified_param)
            tp_names.append('experts.mlp.' + name)

        _pre_dp_module_transform(mb_dmoe.experts.mlp)

        dp_pg = device_mesh['weight_parallel'].get_group(0)
        mb_dmoe.experts = DDP(mb_dmoe.experts, process_group=dp_pg)
        print(f'''
        device_mesh: {device_mesh}
        device_mesh['weight_parallel']: {device_mesh['weight_parallel']}
        device_mesh['expert_parallel']: {device_mesh['expert_parallel']}
        dp_pg: {dp_pg}
        ''')

        # Copy mb_dmoe's parameters to torch_dmoe
        mb_dmoe_state_dict = get_model_state_dict(
            mb_dmoe,
            options=StateDictOptions(full_state_dict=True,),
        )
        for key, t in mb_dmoe_state_dict.items():
            if key in tp_names:
                dtensor_full = DTensor.from_local(
                    t,  # pyright: ignore[reportGeneralTypeIssues]
                    device_mesh=device_mesh,
                    placements=two_d_placements,
                ).full_tensor()

                mb_dmoe_state_dict[key] = dtensor_full
    else:
        mb_dmoe.experts = DDP(mb_dmoe.experts, device_ids=[rank])
        mb_dmoe_state_dict = get_model_state_dict(
            mb_dmoe,
            options=StateDictOptions(full_state_dict=True,),
        )
    mb_dmoe_optimizer = optim.SGD(mb_dmoe.parameters(), lr=0.1)

    # Load mb_dmoe state dict to torch dmoe
    torch_dmoe.module.load_state_dict(mb_dmoe_state_dict, strict=True)

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

if __name__ == "__main__":
    test_dmoe()