'''
https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py
https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
'''

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.tensor.parallel import PrepareModuleInput, SequenceParallel
from torch.distributed._tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel

from src.model import Transformer
from transformers import AutoTokenizer

from src.torch_profiler_utils import get_torch_profiler

# from pdb import set_trace as Tra
from multiprocessing_pdb import MultiprocessingPdb
Tra = MultiprocessingPdb().set_trace


def init_dist():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    print(f"rank: {rank}, world size: {world_size}")
    return rank, world_size

def print_message_with_master_process(rank, message):
    if rank==0:
        print(message)

def test_mesh():
    ################################################################################
    ## init dist setting and model

    rank, world_size = init_dist()
    device = f"cuda:{rank}"

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    block_size = tokenizer.model_max_length
    hidden, nhead, nlayer = 256, 8, 4
    model = Transformer(vocab_size, block_size, hidden, nhead, nlayer).to("cuda")


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
    replicate_group = fsdp_mesh["replicate"].get_group()
    shard_group = fsdp_mesh["shard"].get_group()
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
    # shard_group = fsdp_mesh["shard"].get_group()
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
        ## apply word embedding and loss parallel
        use_loss_parallel = True
        tp_plan = {
            "model.wte": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.ln": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1), # time dimension
                output_layouts=Shard(-1) if use_loss_parallel else Replicate(),
                use_local_output=not use_loss_parallel,
            ),
        }
        parallelize_module(
            model, 
            tp_mesh,
            tp_plan
        )

        ################################################################################
        ## async TP
        enable_async_tp = False
        if enable_async_tp:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group
            torch._inductor.config._micro_pipeline_tp = True
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)
    else:
        use_loss_parallel = False


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
    ## forward and compute loss
    sent = "i love tensor parallelism."
    input_ids = tokenizer(sent, return_tensors='pt').to(device)
    x = input_ids['input_ids']
    labels = input_ids['input_ids'][:, 1:]

    model.train()
    pred = model(x)

    if use_loss_parallel:
        with loss_parallel():
            # assuming pred and labels are of the shape [batch, seq, vocab]
            loss = F.cross_entropy(pred[..., :-1, :].flatten(0, 1), labels.flatten(0, 1))
            loss.backward()
    else:
        loss = F.cross_entropy(pred[..., :-1, :].flatten(0, 1), labels.flatten(0, 1))
        loss.backward()

    print_message_with_master_process(rank, loss)

if __name__ == "__main__":
    test_mesh()