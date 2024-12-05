'''
https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py
https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
https://github.com/facebookresearch/lingua/blob/31b20e172aa9a3e68a73cb501d689291039fc065/lingua/distributed.py#L419-L460
'''

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.tensor.parallel import PrepareModuleInput, SequenceParallel
from torch.distributed._tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel

from transformers import AutoTokenizer
from model import Transformer
from utils import (
    set_seed,
    init_dist,
    print_message_with_master_process,
    _get_torch_dtype,
    cleanup_distributed,
)
from torch_profiler_utils import (
    ContextManagers,
    get_torch_profiler,
)

# from pdb import set_trace as Tra
from multiprocessing_pdb import MultiprocessingPdb
Tra = MultiprocessingPdb().set_trace

USE_TORCH_PROFILER = True
TORCH_PROFILER_LOG_DIR = './assets/torch_profiler_logs'
USE_LOSS_PARALLEL = True
# USE_LOSS_PARALLEL = False


def test_mesh():
    ################################################################################
    ## init dist setting and model

    rank, world_size = init_dist()
    device = f"cuda:{rank}"
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    block_size = tokenizer.model_max_length
    hidden, nhead, nlayer = 1024, 8, 4
    model = Transformer(vocab_size, block_size, hidden, nhead, nlayer).to(device)


    ################################################################################
    ## set mesh for 3d parallel

    ## zero DP + TP for 8 gpu / 1 node
    DP, SHARD, TP = (
        int(os.environ["DP"]), 
        int(os.environ["SHARD"]), 
        int(os.environ["TP"]),
    )
    print_message_with_master_process(rank, f"{world_size} {DP} {SHARD} {TP}")
    assert (world_size == DP*SHARD*TP)

    mesh_3d = init_device_mesh(
        "cuda", 
        (DP, SHARD, TP), 
        mesh_dim_names=("replicate", "shard", "tp"),
    )
    fsdp_mesh = mesh_3d["replicate", "shard"]
    tp_mesh = mesh_3d["tp"]
    replicate_group = mesh_3d["replicate"].get_group()
    shard_group = mesh_3d["shard"].get_group()
    tp_group = tp_mesh.get_group()
    print_message_with_master_process(rank, f'''
    mesh_3d: {mesh_3d}
    fsdp_mesh: {fsdp_mesh}
    tp_mesh: {tp_mesh}
    replicate_group: {replicate_group}
    shard_group: {shard_group}
    tp_group: {tp_group}
    ''')

    # mesh_2d = init_device_mesh(
    #     "cuda", 
    #     (DP*SHARD, TP), 
    #     mesh_dim_names=("shard", "tp"),
    # )
    # fsdp_mesh = mesh_2d["shard"]
    # tp_mesh = mesh_2d["tp"]
    # shard_group = mesh_2d["shard"].get_group()
    # tp_group = tp_mesh.get_group()
    # print_message_with_master_process(rank, f'''
    # mesh_2d: {mesh_2d}
    # fsdp_mesh: {fsdp_mesh}
    # tp_mesh: {tp_mesh}
    # shard_group: {shard_group}
    # tp_group: {tp_group}
    # ''')


    ################################################################################
    ## apply tensor parallel first
    if TP != 1:

        ################################################################################
        ## apply word embedding and loss parallel
        tp_plan = {
            "model.wte": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.ln": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1), # time dimension
                output_layouts=Shard(-1) if USE_LOSS_PARALLEL else Replicate(),
                use_local_output=not USE_LOSS_PARALLEL,
            ),
        }
        parallelize_module(
            model, 
            tp_mesh,
            tp_plan
        )

        ################################################################################
        for layer_id, residual_block in enumerate(model.model.h):
            layer_tp_plan = {
                # Now the input and output of SequenceParallel has Shard(1) layouts,
                # to represent the input/output tensors sharded on the sequence dimension

                ## self attn (sentence parallel -> col parallel -> row parallel)
                "ln1": SequenceParallel(),
                "attn": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attn.q_proj": ColwiseParallel(),
                "attn.k_proj": ColwiseParallel(),
                "attn.v_proj": ColwiseParallel(),
                "attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),

                ## ffn (sentence parallel -> col parallel -> row parallel)
                "ln2": SequenceParallel(),
                "ffn": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                ## im not gonna use GLU, its your turn
                "mlp.ffn1": ColwiseParallel(),
                "mlp.ffn2": RowwiseParallel(output_layouts=Shard(1)),
            }
            ## do parallel
            parallelize_module(
                module=residual_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )

        ################################################################################
        ## async TP
        enable_async_tp = False
        if enable_async_tp:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group
            torch._inductor.config._micro_pipeline_tp = True
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)
    use_loss_parallel_ = USE_LOSS_PARALLEL and (TP != 1)
    print(f'use_loss_parallel_: {use_loss_parallel_}')

    ################################################################################
    ## apply FSDP
    if DP*SHARD != 1:
        model = FSDP(
            model, 
            device_mesh=fsdp_mesh, 
            # use_orig_params=True,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD
        )

        # param_dtype, reduce_dtype = torch.bfloat16, torch.float32
        # mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        # fsdp_config = {"mesh": fsdp_mesh, "mp_policy": mp_policy}
        # if TP != 1:
        #     # check if strided sharding is enabled, which is necessary for 2D/3D DCP
        #     check_strided_sharding_enabled()

        # for layer_id, residual_block in enumerate(model.model.h):
        #     # As an optimization, do not reshard after forward for the last
        #     # transformer block since FSDP would prefetch it immediately
        #     reshard_after_forward = int(layer_id) < len(model.model.h) - 1
        #     fully_shard(
        #         residual_block,
        #         **fsdp_config,
        #         reshard_after_forward=reshard_after_forward,
        #     )
        # fully_shard(model, **fsdp_config, reshard_after_forward=True)

    print_message_with_master_process(rank, f'''
    model: {model}
    ''')

    ################################################################################
    ## define optimizer (it should be compatible with FSDP)
    lr = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)


    ################################################################################
    # get torch profiler
    if USE_TORCH_PROFILER:
        num_wait_steps, num_warmup_steps, num_active_steps, num_repeat = 1, 2, 3, 1
        num_iter = int((num_wait_steps + num_warmup_steps + num_active_steps)*num_repeat)
        context = [
            get_torch_profiler(
                num_wait_steps=num_wait_steps,
                num_warmup_steps=num_warmup_steps,
                num_active_steps=num_active_steps,
                num_repeat=num_repeat,

                root_dir=TORCH_PROFILER_LOG_DIR,
                save_dir_name=f'mesh_test_DP_{DP}_SHARD_{SHARD}_TP_{TP}_USE_LOSS_PARALLEL_{use_loss_parallel_}'
            )
        ]
    else:
        num_iter = 16
        assert num_iter % (batch_size * world_size) == 0
        num_iter //= (batch_size * world_size)
        context = []

    ################################################################################
    ## forward and compute loss
    sent = "i love lingua and 3d parallelism :)"
    input_ids = tokenizer(sent, return_tensors='pt').to(device)
    x = input_ids['input_ids']
    labels = input_ids['input_ids'][:, 1:]
    model.train()

    with ContextManagers(context) as p:
        for i in range(num_iter):

            # forward
            pred = model(x)
            if use_loss_parallel_:
                with loss_parallel():
                    # assuming pred and labels are of the shape [batch, seq, vocab]
                    loss = F.cross_entropy(pred[..., :-1, :].flatten(0, 1), labels.flatten(0, 1))
                    loss.backward()
            else:
                loss = F.cross_entropy(pred[..., :-1, :].flatten(0, 1), labels.flatten(0, 1))
                loss.backward()
            print_message_with_master_process(rank, f'{i}th step loss: {loss}')

            # clear and profile
            optimizer.step()
            optimizer.zero_grad()
            if USE_TORCH_PROFILER:
                p.step()

    cleanup_distributed()

if __name__ == "__main__":
    test_mesh()