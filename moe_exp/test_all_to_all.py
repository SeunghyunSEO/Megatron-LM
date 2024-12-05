'''
https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all

https://github.com/databricks/megablocks/blob/84286de8ab5be0c73928a0059f50c7e2b650e4b1/megablocks/layers/all_to_all.py#L47
https://github.com/databricks/megablocks/blob/84286de8ab5be0c73928a0059f50c7e2b650e4b1/megablocks/layers/moe.py#L404
https://github.com/databricks/megablocks/blob/84286de8ab5be0c73928a0059f50c7e2b650e4b1/megablocks/layers/moe.py#L185
'''
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import (
    set_seed,
    init_dist,
    cleanup_distributed,
    print_message_with_master_process
)

def create_input_for_all_to_all(rank, world_size):
    input = []
    input_splits = []
    
    set_seed(777)
    for rank in range(world_size):
        num_tokens = random.randint(4, 8)
        tokens = torch.arange(rank * 10, rank * 10 + num_tokens, device=f"cuda:{rank}")
        input.append(tokens)
        
        splits = [random.randint(1, max(1, len(tokens) // world_size)) for _ in range(world_size)]
        splits[-1] = len(tokens) - sum(splits[:-1])  # ensure splits add up to the total number of tokens
        input_splits.append(splits)

    return input, input_splits

def test_comm():
    rank, world_size = init_dist(backend="nccl") # backend is NCCL

    input, input_splits = create_input_for_all_to_all(rank, world_size)
    input_splitted = list(input[rank].split(input_splits[rank]))
    output = [torch.empty([item[rank]], dtype=torch.int64, device=f"cuda:{rank}") for item in input_splits]

    print(f'''
    (input) (rank {rank}): {input}
    (input_splits) (rank {rank}): {input_splits}
    (input_splitted) (rank {rank}): {input_splitted}
    (output before all_to_all) (rank {rank}): {output}
    ''')
    dist.all_to_all(
        output, 
        input_splitted,
        group=None,
        async_op=False,
    )
    print(f'''
    (output after all_to_all) (rank {rank}): {output}
    ''')
    dist.barrier()
    print_message_with_master_process(rank, '*****'*20)
    cleanup_distributed()

if __name__ == "__main__":
    test_comm()