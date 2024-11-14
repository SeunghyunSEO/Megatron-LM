import os
import random
import numpy as np
from typing import Optional

import torch
import torch.distributed as dist


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_dist(backend="nccl"):
    rank = int(os.environ["LOCAL_RANK"])
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=backend, init_method="env://", rank=world_rank, world_size=world_size)
    device = torch.device("cuda:{}".format(rank))
    print(f"rank: {rank}, world_rank: {world_rank}, world size: {world_size}, device: {device}")
    
    # https://github.com/facebookresearch/optimizers/blob/89fd01d609019ec5bced42e32be3cdccfa59bab6/distributed_shampoo/examples/trainer_utils.py#L556
    ## wtf, this is so important, if u do not use this, there would be device mismatch 
    torch.cuda.set_device(rank)

    return rank, world_rank, world_size, device

def cleanup_distributed():
    dist.destroy_process_group()

def print_message_with_master_process(rank, message):
    if rank==0:
        print(message)

def _get_torch_dtype(fp16: bool, bf16: bool) -> Optional[torch.dtype]:
    if fp16:
        return torch.float16
    elif bf16:
        return torch.bfloat16
    return None