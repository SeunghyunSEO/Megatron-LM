
## References

- megatron naive moe
    - [Megatron-LM/megatron/core/transformer/moe](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)
    - [Megatron-LM/docs/llama_mistral.md](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/llama_mistral.md)
    - [Megatron-LM/examples/export/ptq_and_trtllm_export](https://github.com/NVIDIA/Megatron-LM/tree/772faca1f8d5030621b738cbd8e8bb2d8d28f6e6/examples/export/ptq_and_trtllm_export)
    - [mixtral inference example](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mixtral)

- dmoe
    - megablocks
        - [moe.py](https://github.com/databricks/megablocks/blob/main/megablocks/layers/moe.py)
        - [dmoe.py](https://github.com/databricks/megablocks/blob/main/megablocks/layers/dmoe.py#L18)
    - llm foundry
        - [layers/ffn.py](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/llmfoundry/models/layers/ffn.py#L470-L509)
        - [test_dmoe.py](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/tests/models/layers/test_dmoe.py#L71)
        - [moe init](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/llmfoundry/models/utils/param_init_fns.py#L341-L404)
    - olmo
        - [olmoe](https://github.com/allenai/OLMo/blob/sewon-olmoe/olmo/model.py#L680-L690)
        - [parallelism](https://github.com/allenai/OLMo/blob/sewon-olmoe/scripts/train.py#L188-L225)
        - [profiler](https://github.com/allenai/OLMo/blob/sewon-olmoe/olmo/train.py#L1225-L1262)

    - megatron integration
        - [megatron PR 1](https://github.com/NVIDIA/Megatron-LM/pull/287)
        - [megatron PR 2](https://github.com/NVIDIA/Megatron-LM/pull/288)
        - [stanford-futuredata/Megatron-LM](https://github.com/stanford-futuredata/Megatron-LM/tree/3a9e3d8de308e6f6398b59d16a8bd7177374f121)

- torch native trainer references
    - [pytorch/torchtune](https://github.com/pytorch/torchtune)
    - [pytorch/torchtitan](https://github.com/pytorch/torchtitan)
        - [memory_profiler](https://github.com/pytorch/torchtitan/blob/main/docs/memory_profiler.md)


## requirements installation

- mcore installation example

```bash
docker run --ipc=host --shm-size=512m --gpus all -it nvcr.io/nvidia/pytorch:24.02-py3
pip install megatron_core
pip install tensorstore==0.1.45
pip install zarr
```

- install megablocks and other dependency

```bash
python -m pip install --upgrade pip

# https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#pip
pip install megablocks==0.6.1

# https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#pip
# pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7
git clone https://github.com/NVIDIA/TransformerEngine
cd TransformerEngine
git checkout release_v1.7
git submodule update --init --recursive
export NVTE_FRAMEWORK=pytorch 
pip install .

# https://github.com/NVIDIA/Megatron-LM/issues/696#issuecomment-1987058741
# https://github.com/NVIDIA/TransformerEngine/issues/1014
# https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
git clone https://github.com/Dao-AILab/flash-attention &&\
cd flash-attention &&\
git checkout v2.5.8 &&\
MAX_JOBS=8 pip install -e .

# https://github.com/fanshiqing/grouped_gemm
pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

```bash
nvcc --version &&\
python -c "import torch; print(torch.__version__); \
import apex; print(apex); \
import transformer_engine as te; print(te); \
print(te.pytorch.Linear); \
import megablocks as mb; print(mb); \
print(mb.layers.dmoe); \
import grouped_gemm; print(grouped_gemm)"
```

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

<details>

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
2.4.0+cu121
<module 'apex' from '/workspace/.local/lib/python3.10/site-packages/apex/__init__.py'>
<module 'transformer_engine' from '/workspace/.local/lib/python3.10/site-packages/transformer_engine/__init__.py'>
<class 'transformer_engine.pytorch.module.linear.Linear'>
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/mlp.py:22: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx: Any, x: torch.Tensor, scale: float):
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/mlp.py:28: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx: torch.Tensor, grad: torch.Tensor):
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/mlp.py:192: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, x, w1, w2, topo, activation_fn):
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/mlp.py:234: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, ddsd_out):
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/mlp.py:402: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, x, w1, w2, batch_sizes, activation_fn):
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/mlp.py:435: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx: Any, ddsd_out: torch.Tensor):
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/glu.py:71: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, x, w1, v1, w2, batch_sizes, activation_fn):
/workspace/.local/lib/python3.10/site-packages/megablocks/layers/glu.py:106: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, ddsd_out):
<module 'megablocks' from '/workspace/.local/lib/python3.10/site-packages/megablocks/__init__.py'>
<module 'megablocks.layers.dmoe' from '/workspace/.local/lib/python3.10/site-packages/megablocks/layers/dmoe.py'>
<module 'grouped_gemm' from '/workspace/.local/lib/python3.10/site-packages/grouped_gemm/__init__.py'>
```

</details>


## wandb

```bash
pip install wandb
wandb login
# python3 -m wandb login
```

## dist pdb tracer 

- Installation

```bash
cd /path/to/dir/multiprocessing_pdb &&\
pip install -e .
```

- Usage

```python
from multiprocessing_pdb import MultiprocessingPdb
Tra = MultiprocessingPdb().set_trace

def dummy_code_block(...):
    Tra()
```


## run scripts for torch/nccl all to all comm

```bash
# cd /path/to/dir/Megatron-LM/moe_exp
export LOCAL_RANK=0 &&\
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23458 &&\
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_all_to_all.py
```


## run scripts for torch native parallelism

```bash
# cd /path/to/dir/Megatron-LM/moe_exp
export LOCAL_RANK=0 &&\
export WORLD_SIZE=2 &&\
export DP=1 &&\
export SHARD=2 &&\
export TP=1 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23458 &&\
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_mesh.py
```

```bash
# cd /path/to/dir/Megatron-LM/moe_exp
export LOCAL_RANK=0 &&\
export WORLD_SIZE=2 &&\
export DP=1 &&\
export SHARD=1 &&\
export TP=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23458 &&\
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_mesh.py
```


## run scripts for dmoe test

```bash
# cd /path/to/dir/Megatron-LM/moe_exp
export LOCAL_RANK=0 &&\
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23458 &&\
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_dmoe.py
```


## download actual training dataset and preprocessing (fineweb or dclm)

```bash
pip install datatrove
```

```bash
# cd /path/to/dir/Megatron-LM/moe_exp
# python download_prepare_hf_data.py dclm_baseline_1.0
python download_prepare_hf_data.py fineweb_edu_10bt
```

```
|-- data
|   |-- fineweb_edu_10bt
|   |   |-- datatrove
|   |   |   |-- completions
|   |   |   |-- executor.json
|   |   |   |-- logs
|   |   |   |-- stats
|   |   |   `-- stats.json
|   |   |-- fineweb_edu_10bt.chunk.00000.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00001.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00002.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00003.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00004.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00005.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00006.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00007.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00008.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00009.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00010.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00011.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00012.jsonl
|   |   |-- fineweb_edu_10bt.chunk.00013.jsonl
|   |   |-- sample
|   |   |   `-- 10BT
|   |   `-- terashuf
|   |       |-- LICENSE
|   |       |-- Makefile
|   |       |-- README.md
|   |       |-- terashuf
|   |       `-- terashuf.cc
|   `-- fineweb_edu_10bt_shuffled
|       |-- fineweb_edu_10bt.chunk.00.jsonl
|       |-- fineweb_edu_10bt.chunk.01.jsonl
|       |-- fineweb_edu_10bt.chunk.02.jsonl
|       |-- fineweb_edu_10bt.chunk.03.jsonl
|       |-- fineweb_edu_10bt.chunk.04.jsonl
|       |-- fineweb_edu_10bt.chunk.05.jsonl
|       |-- fineweb_edu_10bt.chunk.06.jsonl
|       |-- fineweb_edu_10bt.chunk.07.jsonl
|       |-- fineweb_edu_10bt.chunk.08.jsonl
|       |-- fineweb_edu_10bt.chunk.09.jsonl
|       |-- fineweb_edu_10bt.chunk.10.jsonl
|       |-- fineweb_edu_10bt.chunk.11.jsonl
|       |-- fineweb_edu_10bt.chunk.12.jsonl
|       |-- fineweb_edu_10bt.chunk.13.jsonl
|       |-- fineweb_edu_10bt.chunk.14.jsonl
|       |-- fineweb_edu_10bt.chunk.15.jsonl
|       |-- fineweb_edu_10bt.chunk.16.jsonl
|       |-- fineweb_edu_10bt.chunk.17.jsonl
|       |-- fineweb_edu_10bt.chunk.18.jsonl
|       |-- fineweb_edu_10bt.chunk.19.jsonl
|       |-- fineweb_edu_10bt.chunk.20.jsonl
|       |-- fineweb_edu_10bt.chunk.21.jsonl
|       |-- fineweb_edu_10bt.chunk.22.jsonl
|       |-- fineweb_edu_10bt.chunk.23.jsonl
|       |-- fineweb_edu_10bt.chunk.24.jsonl
|       |-- fineweb_edu_10bt.chunk.25.jsonl
|       |-- fineweb_edu_10bt.chunk.26.jsonl
|       |-- fineweb_edu_10bt.chunk.27.jsonl
|       |-- fineweb_edu_10bt.chunk.28.jsonl
|       |-- fineweb_edu_10bt.chunk.29.jsonl
|       |-- fineweb_edu_10bt.chunk.30.jsonl
|       |-- fineweb_edu_10bt.chunk.31.jsonl
|       `-- fineweb_edu_10bt.val.jsonl
```

```bash
$ head -1 fineweb_edu_10bt.val.jsonl
{"text":"C# and the .NET Runtime and Libraries\nIf you are reading this chapter, my guess is that you are interested in learning more about C#. Welcome.\nThis book is primarily about the C# language, but before diving into the details, it is important to understand the basics of the environment in which C# code is written.\nThe C# compiler will take C# programs and convert them into an intermediate language that can be executed only by the .NET Common Language Runtime (CLR). Languages that target a runtime are sometimes known as managed languages1 and are contrasted with unmanaged languages such as C++ that do not require a runtime2 and therefore ...","id":"<urn:uuid:912c3dd3-1d91-44e0-b00f-e7bbed675135>","metadata":{"dump":"CC-MAIN-2019-30","url":"https://www.oreilly.com/library/view/a-programmers-guide/9781430245933/9781430245933_Ch01.xhtml","file_path":"s3://commoncrawl/crawl-data/CC-MAIN-2019-30/segments/1563195529007.88/warc/CC-MAIN-20190723064353-20190723090353-00349.warc.gz","language":"en","language_score":0.9656065106391907,"token_count":135,"score":3.0625,"int_score":3}}
```

```bash
python merge_jsonl_files.py \
--input_dir 'data/fineweb_edu_10bt_shuffled' \
--valid_file_name 'fineweb_edu_10bt.val' \
--output_file_name 'fineweb_edu_10bt_shuffled_merged'
```

```bash
# cd /path/to/dir/Megatron-LM
INPUT="moe_exp/data/fineweb_edu_10bt_shuffled/fineweb_edu_10bt_shuffled_merged.jsonl"
DEST_DIR="moe_exp/data/processed"
mkdir -p $DEST_DIR
OUTPUT_PREFIX="${DEST_DIR}/llama3_fineweb_train"
MERGE_FILE="llama3_fineweb_merges.txt"
TOKENIZER_MODEL_PATH="/path/to/dir/shseo/ckpt/llama3/meta-llama-3.1-8B"
TOKENIZER_TYPE="HuggingFaceTokenizer"
python tools/preprocess_data.py \
--input $INPUT \
--output-prefix $OUTPUT_PREFIX \
--tokenizer-model $TOKENIZER_MODEL_PATH \
--tokenizer-type $TOKENIZER_TYPE \
--merge-file $MERGE_FILE \
--append-eod \
--workers 128
```

```bash
# cd /path/to/dir/Megatron-LM
INPUT="moe_exp/data/fineweb_edu_10bt_shuffled/fineweb_edu_10bt.val.jsonl"
OUTPUT_PREFIX="${DEST_DIR}/llama3_fineweb_eval"
python tools/preprocess_data.py \
--input $INPUT \
--output-prefix $OUTPUT_PREFIX \
--tokenizer-model $TOKENIZER_MODEL_PATH \
--tokenizer-type $TOKENIZER_TYPE \
--merge-file $MERGE_FILE \
--append-eod \
--workers 128
```

## run scripts for megatron

target comparison

```
## dense
Number of parameters in transformer layers in billions:  0.08
Number of parameters in embedding layers in billions: 0.13
Total number of parameters in billions: 0.21

## moe
Number of parameters in transformer layers in billions:  4.24
Number of parameters in embedding layers in billions: 0.13
Total number of parameters in billions: 4.37
```

```
MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 2048
    --position-embedding-type rope
    --rotary-base 100000
    --num-layers 24
    --hidden-size 512
    --num-attention-heads 4
    --group-query-attention
    --num-query-groups 1
    --ffn-hidden-size 1792
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.02
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
)
```

- [tmp](https://github.com/NVIDIA/Megatron-LM/blob/345b1022b80e9653e66ae5bf95a9b3347c72b6a2/megatron/training/theoretical_memory_usage.py#L57)

### dense baseline

```bash
# cd /path/to/dir/Megatron-LM/

export MASTER_ADDR="localhost"
export MASTER_PORT=23456
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0

export USE_MOE=false
export TP=1
export PP=1
export VPP="none"
export EP=1

export LR=0.0006
export MIN_LR=0.00006
export MBSZ=8
export GBSZ=64
export TRAIN_ITERS=40000
export WARMUP_ITERS=1000
export DECAY_ITERS=38000

bash ./moe_exp/scripts/run_moe_from_scratch.sh
```

<details>

```python
Number of parameters in transformer layers in billions:  0.08
Number of parameters in embedding layers in billions: 0.13
Total number of parameters in billions: 0.21
Number of parameters in most loaded shard in billions: 0.2132
Theoretical memory footprints: weight and optimizer=1524.73 MB
[Rank 0] (after 20 iterations) memory (MB) | allocated: 1550.404296875 | max allocated: 20561.49462890625 | reserved: 28404.0 | max reserved: 28404.0
 [2024-10-24 21:02:23] iteration       20/   40000 | consumed samples:         1280 | elapsed time per iteration (ms): 826.6 | throughput per GPU (TFLOP/s/GPU): 23.5 | learning rate: 1.200000E-05 | global batch size:    64 | lm loss: 1.145711E+01 | loss scale: 1.0 | grad norm: 2.456 | num zeros: 57219176.0 | params norm: 315.023 | number of skipped iterations:   0 | number of nan iterations:   0 |
/home/nsml/.local/lib/python3.10/site-packages/torch/distributed/c10d_logger.py:79: FutureWarning: `torch.distributed._all_gather_base` is a private function and will be deprecated. Please use `torch.distributed.all_gather_into_tensor` instead.
  return func(*args, **kwargs)
/path/to/dir/shseo/Megatron-LM/megatron/core/tensor_parallel/layers.py:623: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/path/to/dir/shseo/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:259: FutureWarning: `torch.distributed._reduce_scatter_base` is a private function and will be deprecated. Please use `torch.distributed.reduce_scatter_tensor` instead.
  torch.distributed._reduce_scatter_base(
 [2024-10-24 21:02:27] iteration       40/   40000 | consumed samples:         2560 | elapsed time per iteration (ms): 235.9 | throughput per GPU (TFLOP/s/GPU): 82.4 | learning rate: 2.400000E-05 | global batch size:    64 | lm loss: 1.081154E+01 | loss scale: 1.0 | grad norm: 2.226 | num zeros: 56775512.0 | params norm: 315.029 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:02:32] iteration       60/   40000 | consumed samples:         3840 | elapsed time per iteration (ms): 232.1 | throughput per GPU (TFLOP/s/GPU): 83.8 | learning rate: 3.600000E-05 | global batch size:    64 | lm loss: 1.028284E+01 | loss scale: 1.0 | grad norm: 1.839 | num zeros: 56915840.0 | params norm: 315.094 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:02:37] iteration       80/   40000 | consumed samples:         5120 | elapsed time per iteration (ms): 229.2 | throughput per GPU (TFLOP/s/GPU): 84.8 | learning rate: 4.800000E-05 | global batch size:    64 | lm loss: 9.620361E+00 | loss scale: 1.0 | grad norm: 1.660 | num zeros: 56629560.0 | params norm: 315.315 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:02:42] iteration      100/   40000 | consumed samples:         6400 | elapsed time per iteration (ms): 255.5 | throughput per GPU (TFLOP/s/GPU): 76.1 | learning rate: 6.000000E-05 | global batch size:    64 | lm loss: 8.966025E+00 | loss scale: 1.0 | grad norm: 1.657 | num zeros: 56876176.0 | params norm: 315.817 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:02:46] iteration      120/   40000 | consumed samples:         7680 | elapsed time per iteration (ms): 227.1 | throughput per GPU (TFLOP/s/GPU): 85.6 | learning rate: 7.200000E-05 | global batch size:    64 | lm loss: 8.307431E+00 | loss scale: 1.0 | grad norm: 1.378 | num zeros: 57148496.0 | params norm: 316.607 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:02:51] iteration      140/   40000 | consumed samples:         8960 | elapsed time per iteration (ms): 226.4 | throughput per GPU (TFLOP/s/GPU): 85.9 | learning rate: 8.400000E-05 | global batch size:    64 | lm loss: 7.741142E+00 | loss scale: 1.0 | grad norm: 1.034 | num zeros: 56808672.0 | params norm: 317.693 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:02:55] iteration      160/   40000 | consumed samples:        10240 | elapsed time per iteration (ms): 226.0 | throughput per GPU (TFLOP/s/GPU): 86.0 | learning rate: 9.600000E-05 | global batch size:    64 | lm loss: 7.296268E+00 | loss scale: 1.0 | grad norm: 1.050 | num zeros: 57008768.0 | params norm: 319.004 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:03:00] iteration      180/   40000 | consumed samples:        11520 | elapsed time per iteration (ms): 230.3 | throughput per GPU (TFLOP/s/GPU): 84.4 | learning rate: 1.080000E-04 | global batch size:    64 | lm loss: 6.968433E+00 | loss scale: 1.0 | grad norm: 0.938 | num zeros: 56918588.0 | params norm: 320.358 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 21:03:05] iteration      200/   40000 | consumed samples:        12800 | elapsed time per iteration (ms): 230.2 | throughput per GPU (TFLOP/s/GPU): 84.5 | learning rate: 1.200000E-04 | global batch size:    64 | lm loss: 6.760830E+00 | loss scale: 1.0 | grad norm: 0.659 | num zeros: 56744496.0 | params norm: 321.682 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

</details>

### moe, top-1, 64-experts, 8-way EP

```bash
# cd /path/to/dir/Megatron-LM/

export MASTER_ADDR="localhost"
export MASTER_PORT=23456
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0

export USE_MOE=true
export NUM_EXPERTS=64
export TOPK=1
export MOE_TYPE="moe"

export TP=1
export PP=1
export VPP="none"
export EP=4

export LR=0.0006
export MIN_LR=0.00006
export MBSZ=8
export GBSZ=64
export TRAIN_ITERS=40000
export WARMUP_ITERS=1000
export DECAY_ITERS=38000

bash ./moe_exp/scripts/run_moe_from_scratch.sh
```

<details>

```python
  expert_model_parallel_size ...................... 4
  ffn_hidden_size ................................. 1792    

  moe_aux_loss_coeff .............................. 0.01
  moe_expert_capacity_factor ...................... 1.0
  moe_extended_tp ................................. False
  moe_grouped_gemm ................................ False
  moe_input_jitter_eps ............................ None
  moe_layer_recompute ............................. False
  moe_pad_expert_input_to_capacity ................ True
  moe_per_layer_logging ........................... False
  moe_router_load_balancing_type .................. aux_loss
  moe_router_pre_softmax .......................... True
  moe_router_topk ................................. 1
  moe_shared_expert_intermediate_size ............. None
  moe_shared_expert_overlap ....................... False
  moe_token_dispatcher_type ....................... alltoall
  moe_token_drop_policy ........................... probs
  moe_use_megablocks_dmoe ......................... False
  moe_use_upcycling ............................... False
  moe_z_loss_coeff ................................ None
```

```python
Params for bucket 25 (41287680 elements):
        module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc1.weight
        module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.15.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.12.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.10.linear_fc1.weight
        module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.15.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.12.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.11.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.9.linear_fc1.weight
        module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.13.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc1.weight
        module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc1.weight
        module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.14.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.9.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.14.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.13.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.11.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.10.linear_fc2.weight
        module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc1.weight
        module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.weight
        module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight
Params for bucket 26 (24772608 elements):
        module.decoder.layers.0.mlp.experts.local_experts.8.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.8.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.weight
        module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.weight
        module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc2.weight
```

```python
Number of parameters in transformer layers in billions:  4.24
Number of parameters in embedding layers in billions: 0.13
Total number of parameters in billions: 4.37
Number of parameters in most loaded shard in billions: 4.3750
Theoretical memory footprints: weight and optimizer=31292.23 MB 
[Rank 0] (after 20 iterations) memory (MB) | allocated: 13243.47998046875 | max allocated: 33279.59375 | reserved: 37878.0 | max reserved: 37878.0
 [2024-10-25 03:08:49] iteration       20/   40000 | consumed samples:         1280 | elapsed time per iteration (ms): 1639.0 | throughput per GPU (TFLOP/s/GPU): 11.9 | learning rate: 1.200000E-05 | global batch size:    64 | lm loss: 1.156039E+01 | load_balancing_loss: 1.436231E+00 | loss scale: 1.0 | grad norm: 5.009 | num zeros: 328659928.0 | params norm: 1104.931 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:09:01] iteration       40/   40000 | consumed samples:         2560 | elapsed time per iteration (ms): 628.7 | throughput per GPU (TFLOP/s/GPU): 30.9 | learning rate: 2.400000E-05 | global batch size:    64 | lm loss: 1.093495E+01 | load_balancing_loss: 1.315762E+00 | loss scale: 1.0 | grad norm: 2.197 | num zeros: 642142552.0 | params norm: 1104.930 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:09:14] iteration       60/   40000 | consumed samples:         3840 | elapsed time per iteration (ms): 617.5 | throughput per GPU (TFLOP/s/GPU): 31.5 | learning rate: 3.600000E-05 | global batch size:    64 | lm loss: 1.051737E+01 | load_balancing_loss: 1.099387E+00 | loss scale: 1.0 | grad norm: 2.015 | num zeros: 320483924.0 | params norm: 1104.975 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:09:26] iteration       80/   40000 | consumed samples:         5120 | elapsed time per iteration (ms): 637.7 | throughput per GPU (TFLOP/s/GPU): 30.5 | learning rate: 4.800000E-05 | global batch size:    64 | lm loss: 9.870660E+00 | load_balancing_loss: 1.038399E+00 | loss scale: 1.0 | grad norm: 1.900 | num zeros: 67273898.0 | params norm: 1105.214 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:09:39] iteration      100/   40000 | consumed samples:         6400 | elapsed time per iteration (ms): 622.3 | throughput per GPU (TFLOP/s/GPU): 31.2 | learning rate: 6.000000E-05 | global batch size:    64 | lm loss: 9.142986E+00 | load_balancing_loss: 1.033334E+00 | loss scale: 1.0 | grad norm: 1.922 | num zeros: 84278870.0 | params norm: 1105.793 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:09:51] iteration      120/   40000 | consumed samples:         7680 | elapsed time per iteration (ms): 633.8 | throughput per GPU (TFLOP/s/GPU): 30.7 | learning rate: 7.200000E-05 | global batch size:    64 | lm loss: 8.465843E+00 | load_balancing_loss: 1.020885E+00 | loss scale: 1.0 | grad norm: 1.439 | num zeros: 62977510.0 | params norm: 1106.440 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:10:04] iteration      140/   40000 | consumed samples:         8960 | elapsed time per iteration (ms): 631.6 | throughput per GPU (TFLOP/s/GPU): 30.8 | learning rate: 8.400000E-05 | global batch size:    64 | lm loss: 7.900618E+00 | load_balancing_loss: 1.013641E+00 | loss scale: 1.0 | grad norm: 1.037 | num zeros: 59860808.0 | params norm: 1107.035 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:10:16] iteration      160/   40000 | consumed samples:        10240 | elapsed time per iteration (ms): 615.5 | throughput per GPU (TFLOP/s/GPU): 31.6 | learning rate: 9.600000E-05 | global batch size:    64 | lm loss: 7.463532E+00 | load_balancing_loss: 1.011339E+00 | loss scale: 1.0 | grad norm: 23.285 | num zeros: 60071364.0 | params norm: 1107.852 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:10:29] iteration      180/   40000 | consumed samples:        11520 | elapsed time per iteration (ms): 629.1 | throughput per GPU (TFLOP/s/GPU): 30.9 | learning rate: 1.080000E-04 | global batch size:    64 | lm loss: 7.170358E+00 | load_balancing_loss: 1.014433E+00 | loss scale: 1.0 | grad norm: 0.773 | num zeros: 59741638.0 | params norm: 1108.843 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:10:41] iteration      200/   40000 | consumed samples:        12800 | elapsed time per iteration (ms): 618.3 | throughput per GPU (TFLOP/s/GPU): 31.4 | learning rate: 1.200000E-04 | global batch size:    64 | lm loss: 6.977175E+00 | load_balancing_loss: 1.013544E+00 | loss scale: 1.0 | grad norm: 0.446 | num zeros: 59918562.0 | params norm: 1110.084 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:10:53] iteration      220/   40000 | consumed samples:        14080 | elapsed time per iteration (ms): 599.8 | throughput per GPU (TFLOP/s/GPU): 32.4 | learning rate: 1.320000E-04 | global batch size:    64 | lm loss: 6.759798E+00 | load_balancing_loss: 1.010501E+00 | loss scale: 1.0 | grad norm: 0.787 | num zeros: 59833985.0 | params norm: 1111.568 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:11:06] iteration      240/   40000 | consumed samples:        15360 | elapsed time per iteration (ms): 632.5 | throughput per GPU (TFLOP/s/GPU): 30.7 | learning rate: 1.440000E-04 | global batch size:    64 | lm loss: 6.588043E+00 | load_balancing_loss: 1.012835E+00 | loss scale: 1.0 | grad norm: 0.492 | num zeros: 60148859.0 | params norm: 1113.448 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:11:19] iteration      260/   40000 | consumed samples:        16640 | elapsed time per iteration (ms): 634.9 | throughput per GPU (TFLOP/s/GPU): 30.6 | learning rate: 1.560000E-04 | global batch size:    64 | lm loss: 6.426376E+00 | load_balancing_loss: 1.010556E+00 | loss scale: 1.0 | grad norm: 0.470 | num zeros: 59930712.0 | params norm: 1115.626 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:11:31] iteration      280/   40000 | consumed samples:        17920 | elapsed time per iteration (ms): 605.6 | throughput per GPU (TFLOP/s/GPU): 32.1 | learning rate: 1.680000E-04 | global batch size:    64 | lm loss: 6.265887E+00 | load_balancing_loss: 1.008574E+00 | loss scale: 1.0 | grad norm: 0.469 | num zeros: 59878548.0 | params norm: 1118.098 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-25 03:11:43] iteration      300/   40000 | consumed samples:        19200 | elapsed time per iteration (ms): 621.4 | throughput per GPU (TFLOP/s/GPU): 31.3 | learning rate: 1.800000E-04 | global batch size:    64 | lm loss: 6.147610E+00 | load_balancing_loss: 1.009638E+00 | loss scale: 1.0 | grad norm: 0.555 | num zeros: 59985277.0 | params norm: 1120.943 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

</details>


### dropless moe, top-1, 64-experts, 8-way EP

```bash
# cd /path/to/dir/Megatron-LM/

export MASTER_ADDR="localhost"
export MASTER_PORT=23456
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0

export USE_MOE=true
export NUM_EXPERTS=64
export TOPK=1
export MOE_TYPE="dmoe"

export TP=1
export PP=1
export VPP="none"
# export EP=8
export EP=4

export LR=0.0006
export MIN_LR=0.00006
export MBSZ=8
export GBSZ=64
export TRAIN_ITERS=40000
export WARMUP_ITERS=1000
export DECAY_ITERS=38000

bash ./moe_exp/scripts/run_moe_from_scratch.sh
```


- EP 4 / accum 1

<details>

```python
DISTRIBUTED_ARGS: --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 23456

MODEL_ARGS: --disable-bias-linear --seq-length 2048 --max-position-embeddings 2048 --position-embedding-type rope --rotary-base 100000 --num-layers 24 --hidden-size 512 --num-attention-heads 4 --group-query-attention --num-query-groups 1 --ffn-hidden-siz
e 1792 --swiglu --normalization RMSNorm --init-method-std 0.02 --untie-embeddings-and-output-weights --attention-dropout 0.0 --hidden-dropout 0.0

MOE_ARGS: --num-experts 64 --moe-router-topk 1 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2 --moe-token-dispatcher-type alltoall --moe-router-pre-softmax --moe-grouped-gemm

MODEL_PARALLEL_ARGS: --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --sequence-parallel --use-distributed-optimizer
DATA_ARGS: --tokenizer-type HuggingFaceTokenizer --tokenizer-model /path/to/dir/shseo/ckpt/llama3/meta-llama-3.1-8B --train-data-path /path/to/dir/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document --valid-data-path /path/to/dir/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document

TRAINING_ARGS: --micro-batch-size 8 --global-batch-size 64 --train-iters 40000 --lr-decay-iters 38000 --lr-warmup-iters 1000 --lr 0.0006 --min-lr 0.00006 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0 --bf16 --use-flash-attn --accumulate-allreduce-grads-in-fp32 --overlap-grad-reduce --overlap-param-gather --no-gradient-accumulation-fusion

SAVING_ARGS: --save /path/to/dir/shseo/checkpoint/megatron/moe_true_num_experts_64_topk_1_type_dmoe_num_nodes_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_4 --load /path/to/dir/shseo/checkpoint/megatron/moe_true_num_experts_64_topk_1_type_dmoe_num_nodes_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_4 --save-interval 1000000

LOGGING_ARGS: --log-interval 20 --log-throughput --log-params-norm --log-num-zeros-in-grad --log-progress --eval-interval 1000 --eval-iters 10 --tensorboard-dir /path/to/dir/shseo/checkpoint/megatron/moe_true_num_experts_64_topk_1_type_dmoe_num_nodes
_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_4/tensorboard --tensorboard-log-interval 20 --log-timers-to-tensorboard --log-memory-to-tensorboard --log-validation-ppl-to-tensorboard --log-world-size-to-tensorboard --no-load-optim --no-load-rng --wandb-project megatron --wandb-exp-name moe_true_num_experts_64_topk_1_type_dmoe_num_nodes_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_4    
```

```python
using world size: 8, data-parallel size: 8, context-parallel size: 1, tensor-model-parallel size: 1, encoder-tensor-model-parallel size: 0, pipeline-model-parallel size: 1, encoder-pipeline-model-parallel size: 0                                          
WARNING: overriding default arguments for tokenizer_type:GPT2BPETokenizer                        with tokenizer_type:HuggingFaceTokenizer 

  moe_aux_loss_coeff .............................. 0.01
  moe_expert_capacity_factor ...................... None
  moe_extended_tp ................................. False
  moe_grouped_gemm ................................ True
  moe_input_jitter_eps ............................ None
  moe_layer_recompute ............................. False
  moe_pad_expert_input_to_capacity ................ False
  moe_per_layer_logging ........................... False
  moe_router_load_balancing_type .................. aux_loss
  moe_router_pre_softmax .......................... True
  moe_router_topk ................................. 1
  moe_shared_expert_intermediate_size ............. None
  moe_shared_expert_overlap ....................... False
  moe_token_dispatcher_type ....................... alltoall
  moe_token_drop_policy ........................... probs
  moe_use_megablocks_dmoe ......................... False
  moe_use_upcycling ............................... False
  moe_z_loss_coeff ................................ None

  expert_model_parallel_size ...................... 4

  num_attention_heads ............................. 4
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_dataset_builder_threads ..................... 1
  num_experts ..................................... 64
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_query_groups ................................ 1  

  overlap_grad_reduce ............................. True
  overlap_p2p_comm ................................ False
  overlap_param_gather ............................ True
  overlap_param_gather_with_optimizer_step ........ False
  override_opt_param_scheduler .................... False

  params_dtype .................................... torch.bfloat16
  patch_dim ....................................... 16   

  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ None 

  tp_comm_bootstrap_backend ....................... nccl
  tp_comm_bulk_dgrad .............................. True
  tp_comm_bulk_wgrad .............................. True
  tp_comm_overlap ................................. False
  tp_comm_overlap_ag .............................. True
  tp_comm_overlap_cfg ............................. None
  tp_comm_overlap_rs .............................. True
  tp_comm_overlap_rs_dgrad ........................ False
  tp_comm_split_ag ................................ True
  tp_comm_split_rs ................................ True   
```

```python
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 1204838912
INFO:megatron.core.distributed.distributed_data_parallel:Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=True, overlap_param_gather=True, align_param_gather=False, use_distributed
_optimizer=True, check_for_nan_in_grad=True, bucket_size=40000000, average_in_collective=False, fp8_param_gather=False)
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 2
Params for bucket 1 (65667072 elements):
        module.output_layer.weight
Params for bucket 2 (82207232 elements):
        module.decoder.layers.22.mlp.router.weight
        module.decoder.layers.18.mlp.router.weight
        module.decoder.layers.15.self_attention.linear_qkv.weight
        module.decoder.layers.13.pre_mlp_layernorm.weight
        module.decoder.layers.4.self_attention.linear_proj.weight
        module.decoder.layers.1.mlp.router.weight
        module.decoder.layers.0.pre_mlp_layernorm.weight
        module.decoder.layers.23.self_attention.linear_proj.weight
        module.decoder.layers.19.self_attention.linear_qkv.weight
        module.decoder.layers.9.self_attention.linear_proj.weight
        module.decoder.layers.8.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.3.self_attention.linear_qkv.weight
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 24
Params for bucket 1 (44040192 elements):
        module.decoder.layers.23.mlp.experts.weight2
        module.decoder.layers.23.mlp.experts.weight1
Params for bucket 2 (44040192 elements):

...

Params for bucket 12 (44040192 elements):
        module.decoder.layers.12.mlp.experts.weight2
        module.decoder.layers.12.mlp.experts.weight1
Params for bucket 13 (44040192 elements):
        module.decoder.layers.11.mlp.experts.weight2
        module.decoder.layers.11.mlp.experts.weight1
Params for bucket 14 (44040192 elements):
        module.decoder.layers.10.mlp.experts.weight2
        module.decoder.layers.10.mlp.experts.weight1
Params for bucket 15 (44040192 elements):
        module.decoder.layers.9.mlp.experts.weight2
        module.decoder.layers.9.mlp.experts.weight1
Params for bucket 16 (44040192 elements):
        module.decoder.layers.8.mlp.experts.weight2
        module.decoder.layers.8.mlp.experts.weight1
Params for bucket 17 (44040192 elements):
        module.decoder.layers.7.mlp.experts.weight2
        module.decoder.layers.7.mlp.experts.weight1
Params for bucket 18 (44040192 elements):
        module.decoder.layers.6.mlp.experts.weight2
        module.decoder.layers.6.mlp.experts.weight1
Params for bucket 19 (44040192 elements):
        module.decoder.layers.5.mlp.experts.weight2
        module.decoder.layers.5.mlp.experts.weight1
Params for bucket 20 (44040192 elements):
        module.decoder.layers.4.mlp.experts.weight2
        module.decoder.layers.4.mlp.experts.weight1
Params for bucket 21 (44040192 elements):
        module.decoder.layers.3.mlp.experts.weight2
        module.decoder.layers.3.mlp.experts.weight1
Params for bucket 22 (44040192 elements):
        module.decoder.layers.2.mlp.experts.weight2
        module.decoder.layers.2.mlp.experts.weight1
Params for bucket 23 (44040192 elements):
        module.decoder.layers.1.mlp.experts.weight2
        module.decoder.layers.1.mlp.experts.weight1
Params for bucket 24 (44040192 elements):
        module.decoder.layers.0.mlp.experts.weight2
        module.decoder.layers.0.mlp.experts.weight1              
```

```python
INFO:megatron.core.datasets.blended_megatron_dataset_builder:Building dataset splits with cls=GPTDataset, sizes=(2560000, 26240, 640), and config=GPTDatasetConfig(random_seed=1234, sequence_length=2048, blend=None, blend_per_split=[(['/path/to/dir/sh
seo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document'], None), (['/path/to/dir/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document'], None), None], renormalize_blend_weights=False, split=None, split_matrix=N
one, num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True, mock=False, tokenizer=<megatron.training.tokenizer.tokenizer._HuggingFaceTokenizer object at 0x7effd0397820>, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss
=False, create_attention_mask=True, drop_last_partial_validation_sequence=True, add_extra_token_to_sequence=True, s3_cache_path=None)
INFO:megatron.core.datasets.indexed_dataset:Load the _IndexReader from /path/to/dir/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document.idx
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence lengths
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence pointers
INFO:megatron.core.datasets.indexed_dataset:    Extract the document indices
INFO:megatron.core.datasets.indexed_dataset:> total number of sequences: 320000
INFO:megatron.core.datasets.indexed_dataset:> total number of documents: 320000
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset train indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from 1880e59ae53240a6c99e2e3f147ab535-GPTDataset-train-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from 1880e59ae53240a6c99e2e3f147ab535-GPTDataset-train-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from 1880e59ae53240a6c99e2e3f147ab535-GPTDataset-train-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 2662358
INFO:megatron.core.datasets.indexed_dataset:Load the _IndexReader from /path/to/dir/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document.idx
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence lengths
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence pointers
INFO:megatron.core.datasets.indexed_dataset:    Extract the document indices
INFO:megatron.core.datasets.indexed_dataset:> total number of sequences: 320000
INFO:megatron.core.datasets.indexed_dataset:> total number of documents: 320000
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset valid indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from be42cf73af7730d632dfd37ba62e6f81-GPTDataset-valid-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from be42cf73af7730d632dfd37ba62e6f81-GPTDataset-valid-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from be42cf73af7730d632dfd37ba62e6f81-GPTDataset-valid-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 156609
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2024-10-24 16:21:50
done with setup ...
(min, max) time across ranks (ms):                             
    model-and-optimizer-setup ......................: (245.01, 247.40)                                                         
    train/valid/test-data-iterators-setup ..........: (171.11, 179.28)training ...  
```

```python
Number of parameters in transformer layers in billions:  4.24                                                                  
Number of parameters in embedding layers in billions: 0.13                                                                     
Total number of parameters in billions: 4.37                   
Number of parameters in most loaded shard in billions: 4.3750                                                                  
Theoretical memory footprints: weight and optimizer=31292.23 MB                                                                
[Rank 0] (after 20 iterations) memory (MB) | allocated: 13182.66748046875 | max allocated: 33742.08056640625 | reserved: 37658.0 | max reserved: 37658.0                                                                                                      
 [2024-10-24 16:22:31] iteration       20/   40000 | consumed samples:         1280 | elapsed time per iteration (ms): 2027.3 | throughput per GPU (TFLOP/s/GPU): 9.6 | learning rate: 1.200000E-05 | global batch size:    64 | lm loss: 1.155841E+01 | load_balancing_loss: 1.436934E+00 | loss scale: 1.0 | grad norm: 4.951 | num zeros: 312072984.0 | params norm: 1104.963 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:22:41] iteration       40/   40000 | consumed samples:         2560 | elapsed time per iteration (ms): 494.3 | throughput per GPU (TFLOP/s/GPU): 39.3 | learning rate: 2.400000E-05 | global batch size:    64 | lm loss: 1.093323E+01 | load_balancing_loss: 1.314004E+00 | loss scale: 1.0 | grad norm: 2.178 | num zeros: 647600308.0 | params norm: 1104.959 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:22:51] iteration       60/   40000 | consumed samples:         3840 | elapsed time per iteration (ms): 515.1 | throughput per GPU (TFLOP/s/GPU): 37.8 | learning rate: 3.600000E-05 | global batch size:    64 | lm loss: 1.051320E+01 | load_balancing_loss: 1.100239E+00 | loss scale: 1.0 | grad norm: 2.012 | num zeros: 248852404.0 | params norm: 1105.003 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:23:01] iteration       80/   40000 | consumed samples:         5120 | elapsed time per iteration (ms): 524.7 | throughput per GPU (TFLOP/s/GPU): 37.1 | learning rate: 4.800000E-05 | global batch size:    64 | lm loss: 9.843142E+00 | load_balancing_loss: 1.045322E+00 | loss scale: 1.0 | grad norm: 2.116 | num zeros: 70030195.0 | params norm: 1105.230 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:23:12] iteration      100/   40000 | consumed samples:         6400 | elapsed time per iteration (ms): 523.8 | throughput per GPU (TFLOP/s/GPU): 37.1 | learning rate: 6.000000E-05 | global batch size:    64 | lm loss: 9.108113E+00 | load_balancing_loss: 1.032283E+00 | loss scale: 1.0 | grad norm: 1.846 | num zeros: 65071031.0 | params norm: 1105.731 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:23:22] iteration      120/   40000 | consumed samples:         7680 | elapsed time per iteration (ms): 519.1 | throughput per GPU (TFLOP/s/GPU): 37.5 | learning rate: 7.200000E-05 | global batch size:    64 | lm loss: 8.429337E+00 | load_balancing_loss: 1.023230E+00 | loss scale: 1.0 | grad norm: 1.522 | num zeros: 60118642.0 | params norm: 1106.295 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:23:33] iteration      140/   40000 | consumed samples:         8960 | elapsed time per iteration (ms): 522.3 | throughput per GPU (TFLOP/s/GPU): 37.2 | learning rate: 8.400000E-05 | global batch size:    64 | lm loss: 7.871146E+00 | load_balancing_loss: 1.015670E+00 | loss scale: 1.0 | grad norm: 1.042 | num zeros: 59781933.0 | params norm: 1106.910 | number of skipped iterations:   0 | number of nan iterations:   0 |

 [2024-10-24 16:30:14] iteration     1040/   40000 | consumed samples:        66560 | elapsed time per iteration (ms): 439.3 | throughput per GPU (TFLOP/s/GPU): 44.3 | learning rate: 5.999984E-04 | global batch size:    64 | lm loss: 4.441610E+00 | load_balancing_loss: 1.002532E+00 | loss scale: 1.0 | grad norm: 3.184 | num zeros: 60338724.0 | params norm: 1377.142 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:30:23] iteration     1060/   40000 | consumed samples:        67840 | elapsed time per iteration (ms): 437.2 | throughput per GPU (TFLOP/s/GPU): 44.5 | learning rate: 5.999965E-04 | global batch size:    64 | lm loss: 4.419007E+00 | load_balancing_loss: 1.000721E+00 | loss scale: 1.0 | grad norm: 0.331 | num zeros: 60389132.0 | params norm: 1387.913 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:30:32] iteration     1080/   40000 | consumed samples:        69120 | elapsed time per iteration (ms): 417.8 | throughput per GPU (TFLOP/s/GPU): 46.5 | learning rate: 5.999938E-04 | global batch size:    64 | lm loss: 4.361573E+00 | load_balancing_loss: 9.994035E-01 | loss scale: 1.0 | grad norm: 1.114 | num zeros: 60137894.0 | params norm: 1397.674 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:30:40] iteration     1100/   40000 | consumed samples:        70400 | elapsed time per iteration (ms): 416.2 | throughput per GPU (TFLOP/s/GPU): 46.7 | learning rate: 5.999903E-04 | global batch size:    64 | lm loss: 4.337642E+00 | load_balancing_loss: 9.981837E-01 | loss scale: 1.0 | grad norm: 0.332 | num zeros: 60598042.0 | params norm: 1406.977 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:30:48] iteration     1120/   40000 | consumed samples:        71680 | elapsed time per iteration (ms): 416.3 | throughput per GPU (TFLOP/s/GPU): 46.7 | learning rate: 5.999860E-04 | global batch size:    64 | lm loss: 4.312704E+00 | load_balancing_loss: 9.967974E-01 | loss scale: 1.0 | grad norm: 0.387 | num zeros: 60441200.0 | params norm: 1416.001 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 16:30:57] iteration     1140/   40000 | consumed samples:        72960 | elapsed time per iteration (ms): 421.4 | throughput per GPU (TFLOP/s/GPU): 46.1 | learning rate: 5.999809E-04 | global batch size:    64 | lm loss: 4.289932E+00 | load_balancing_loss: 9.977578E-01 | loss scale: 1.0 | grad norm: 0.366 | num zeros: 60324591.0 | params norm: 1425.171 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

- reserved (not accurate because it's nvidia-smi)

```python
+-----------------------------------------+----------------------+----------------------+
|   5  NVIDIA A100-SXM4-80GB          On  | 00000000:CA:00.0 Off |                  Off |
| N/A   54C    P0             201W / 400W |  45634MiB / 81920MiB |     97%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   6  NVIDIA A100-SXM4-80GB          On  | 00000000:E3:00.0 Off |                  Off |
| N/A   51C    P0             207W / 400W |  45866MiB / 81920MiB |     91%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   7  NVIDIA A100-SXM4-80GB          On  | 00000000:E7:00.0 Off |                  Off |
| N/A   59C    P0             204W / 400W |  45368MiB / 81920MiB |     97%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
```

</details>


- EP 8 / accum 2


<details>

```python
DISTRIBUTED_ARGS: --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 23456

MODEL_ARGS: --disable-bias-linear --seq-length 2048 --max-position-embeddings 2048 --position-embedding-type rope --rotary-base 100000 --num-layers 24 --hidden-size 512 --num-attention-heads 4 --group-query-attention --num-query-groups 1 --ffn-hidden-size 1792 --swiglu --normalization RMSNorm --init-method-std 0.02 --untie-embeddings-and-output-weights --attention-dropout 0.0 --hidden-dropout 0.0

MOE_ARGS: --num-experts 64 --moe-router-topk 1 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2 --moe-token-dispatcher-type alltoall --moe-router-pre-softmax --moe-grouped-gemm

MODEL_PARALLEL_ARGS: --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 8 --sequence-parallel --use-distributed-optimizer
DATA_ARGS: --tokenizer-type HuggingFaceTokenizer --tokenizer-model /path/to/dir/shseo/ckpt/llama3/meta-llama-3.1-8B --train-data-path /path/to/dir/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document --valid-data-path /path/to/dir/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document

TRAINING_ARGS: --micro-batch-size 4 --global-batch-size 64 --train-iters 40000 --lr-decay-iters 38000 --lr-warmup-iters 1000 --lr 0.0006 --min-lr 0.00006 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0 --bf16 --use-flash-attn --accumulate-allreduce-grads-in-fp32 --overlap-grad-reduce --overlap-param-gather --no-gradient-accumulation-fusion

SAVING_ARGS: --save /path/to/dir/shseo/checkpoint/megatron/moe_true_num_experts_64_topk_1_type_dmoe_num_nodes_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_8 --load /path/to/dir/shseo/checkpoint/megatron/moe_true_num_experts_64_topk_1_type_dmoe_num_nodes_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_8 --use-dist-ckpt --auto-detect-ckpt-format --dist-ckpt-format zarr --ckpt-fully-parallel-save --use-checkpoint-opt_param-scheduler

LOGGING_ARGS: --log-interval 20 --log-throughput --log-params-norm --log-num-zeros-in-grad --log-progress --save-interval 10000 --eval-interval 1000 --eval-iters 10 --tensorboard-dir /path/to/dir/shseo/checkpoint/megatron/moe_true_num_experts_64_topk_1_type_dmoe_num_nodes_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_8/tensorboard --tensorboard-log-interval 20 --log-timers-to-tensorboard --log-memory-to-tensorboard --log-validation-ppl-to-tensorboard --log-world-size-to-tensorboard --no-load-optim --no-load-rng --wandb-project megatron --wandb-exp-name moe_true_num_experts_64_topk_1_type_dmoe_num_nodes_1_gpus_per_node_8_TP_1_PP_1_VPP_none_EP_8
```


```python
  moe_aux_loss_coeff .............................. 0.01
  moe_expert_capacity_factor ...................... None
  moe_extended_tp ................................. False
  moe_grouped_gemm ................................ True
  moe_input_jitter_eps ............................ None
  moe_layer_recompute ............................. False
  moe_pad_expert_input_to_capacity ................ False
  moe_per_layer_logging ........................... False
  moe_router_load_balancing_type .................. aux_loss
  moe_router_pre_softmax .......................... True
  moe_router_topk ................................. 1
  moe_shared_expert_intermediate_size ............. None
  moe_shared_expert_overlap ....................... False
  moe_token_dispatcher_type ....................... alltoall
  moe_token_drop_policy ........................... probs
  moe_use_megablocks_dmoe ......................... False
  moe_use_upcycling ............................... False
  moe_z_loss_coeff ................................ None
```

```python
INFO:megatron.core.distributed.distributed_data_parallel:Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=True, overlap_param_gather=True, align_param_gather=False, use_distributed
_optimizer=True, check_for_nan_in_grad=True, bucket_size=40000000, average_in_collective=False, fp8_param_gather=False)        
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 2             
Params for bucket 1 (65667072 elements):
        module.output_layer.weight
Params for bucket 2 (82207232 elements):
        module.decoder.layers.5.self_attention.linear_qkv.weight
        module.decoder.layers.9.self_attention.linear_proj.weight
        module.decoder.layers.4.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.3.pre_mlp_layernorm.weight
        module.decoder.layers.1.pre_mlp_layernorm.weight
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.final_layernorm.weight
        module.decoder.layers.10.mlp.router.weight
        module.decoder.layers.6.mlp.router.weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.21.self_attention.linear_proj.weight
        module.decoder.layers.13.self_attention.linear_proj.weight
        module.decoder.layers.12.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.23.pre_mlp_layernorm.weight
        module.decoder.layers.22.pre_mlp_layernorm.weight
        module.decoder.layers.21.mlp.router.weight
        module.decoder.layers.15.pre_mlp_layernorm.weight
        module.decoder.layers.7.pre_mlp_layernorm.weight
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 12   

Params for bucket 9 (44040192 elements):
        module.decoder.layers.7.mlp.experts.weight2
        module.decoder.layers.7.mlp.experts.weight1
        module.decoder.layers.6.mlp.experts.weight2
        module.decoder.layers.6.mlp.experts.weight1
Params for bucket 10 (44040192 elements):
        module.decoder.layers.5.mlp.experts.weight2
        module.decoder.layers.5.mlp.experts.weight1
        module.decoder.layers.4.mlp.experts.weight2
        module.decoder.layers.4.mlp.experts.weight1
Params for bucket 11 (44040192 elements):
        module.decoder.layers.3.mlp.experts.weight2
        module.decoder.layers.2.mlp.experts.weight2
        module.decoder.layers.2.mlp.experts.weight1
        module.decoder.layers.3.mlp.experts.weight1
Params for bucket 12 (44040192 elements):
        module.decoder.layers.1.mlp.experts.weight2
        module.decoder.layers.1.mlp.experts.weight1
        module.decoder.layers.0.mlp.experts.weight2
        module.decoder.layers.0.mlp.experts.weight1 
```

```python
Number of parameters in transformer layers in billions:  4.24
Number of parameters in embedding layers in billions: 0.13
Total number of parameters in billions: 4.37
Number of parameters in most loaded shard in billions: 4.3750
Theoretical memory footprints: weight and optimizer=31292.23 MB
[Rank 0] (after 20 iterations) memory (MB) | allocated: 10158.16748046875 | max allocated: 20612.236328125 | reserved: 24654.0 | max reserved: 24654.0
 [2024-10-24 09:49:45] iteration       20/   40000 | consumed samples:         1280 | elapsed time per iteration (ms): 3367.6 | throughput per GPU (TFLOP/s/GPU): 5.8 | learning rate: 1.200000E-05 | global batch size:    64 | lm loss: 1.155786E+01 | load_balancing_loss: 1.534522E+00 | loss scale: 1.0 | grad norm: 4.963 | num zeros: 345108692.0 | params norm: 1104.962 | number of skipped iterations:   0 | number of nan iterations:   0 |
/home/nsml/.local/lib/python3.10/site-packages/torch/distributed/c10d_logger.py:79: FutureWarning: `torch.distributed._all_gather_base` is a private function and will be deprecated. Please use `torch.distributed.all_gather_into_tensor` instead.
  return func(*args, **kwargs)
/path/to/dir/shseo/Megatron-LM/megatron/core/tensor_parallel/layers.py:623: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/path/to/dir/shseo/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:259: FutureWarning: `torch.distributed._reduce_scatter_base` is a private function and will be deprecated. Please use `torch.distributed.reduce_scatter_tensor` instead.
  torch.distributed._reduce_scatter_base(
 [2024-10-24 09:50:19] iteration       40/   40000 | consumed samples:         2560 | elapsed time per iteration (ms): 1688.1 | throughput per GPU (TFLOP/s/GPU): 11.5 | learning rate: 2.400000E-05 | global batch size:    64 | lm loss: 1.093352E+01 | load_balancing_loss: 1.321269E+00 | loss scale: 1.0 | grad norm: 2.191 | num zeros: 675105940.0 | params norm: 1104.958 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:50:51] iteration       60/   40000 | consumed samples:         3840 | elapsed time per iteration (ms): 1646.4 | throughput per GPU (TFLOP/s/GPU): 11.8 | learning rate: 3.600000E-05 | global batch size:    64 | lm loss: 1.051435E+01 | load_balancing_loss: 1.101528E+00 | loss scale: 1.0 | grad norm: 1.998 | num zeros: 301178424.0 | params norm: 1105.003 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:51:24] iteration       80/   40000 | consumed samples:         5120 | elapsed time per iteration (ms): 1646.4 | throughput per GPU (TFLOP/s/GPU): 11.8 | learning rate: 4.800000E-05 | global batch size:    64 | lm loss: 9.842927E+00 | load_balancing_loss: 1.047027E+00 | loss scale: 1.0 | grad norm: 1.915 | num zeros: 72867846.0 | params norm: 1105.228 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:51:57] iteration      100/   40000 | consumed samples:         6400 | elapsed time per iteration (ms): 1623.0 | throughput per GPU (TFLOP/s/GPU): 12.0 | learning rate: 6.000000E-05 | global batch size:    64 | lm loss: 9.109857E+00 | load_balancing_loss: 1.036356E+00 | loss scale: 1.0 | grad norm: 1.894 | num zeros: 73314939.0 | params norm: 1105.727 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:52:29] iteration      120/   40000 | consumed samples:         7680 | elapsed time per iteration (ms): 1592.3 | throughput per GPU (TFLOP/s/GPU): 12.2 | learning rate: 7.200000E-05 | global batch size:    64 | lm loss: 8.423096E+00 | load_balancing_loss: 1.025098E+00 | loss scale: 1.0 | grad norm: 1.458 | num zeros: 68378734.0 | params norm: 1106.307 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:53:00] iteration      140/   40000 | consumed samples:         8960 | elapsed time per iteration (ms): 1554.9 | throughput per GPU (TFLOP/s/GPU): 12.5 | learning rate: 8.400000E-05 | global batch size:    64 | lm loss: 7.857308E+00 | load_balancing_loss: 1.018348E+00 | loss scale: 1.0 | grad norm: 1.521 | num zeros: 59453011.0 | params norm: 1106.928 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:53:31] iteration      160/   40000 | consumed samples:        10240 | elapsed time per iteration (ms): 1547.8 | throughput per GPU (TFLOP/s/GPU): 12.6 | learning rate: 9.600000E-05 | global batch size:    64 | lm loss: 7.449809E+00 | load_balancing_loss: 1.020225E+00 | loss scale: 1.0 | grad norm: 0.597 | num zeros: 60060012.0 | params norm: 1107.684 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:54:01] iteration      180/   40000 | consumed samples:        11520 | elapsed time per iteration (ms): 1531.6 | throughput per GPU (TFLOP/s/GPU): 12.7 | learning rate: 1.080000E-04 | global batch size:    64 | lm loss: 7.160819E+00 | load_balancing_loss: 1.016319E+00 | loss scale: 1.0 | grad norm: 0.537 | num zeros: 59998068.0 | params norm: 1108.624 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:54:32] iteration      200/   40000 | consumed samples:        12800 | elapsed time per iteration (ms): 1532.0 | throughput per GPU (TFLOP/s/GPU): 12.7 | learning rate: 1.200000E-04 | global batch size:    64 | lm loss: 6.963683E+00 | load_balancing_loss: 1.015341E+00 | loss scale: 1.0 | grad norm: 0.439 | num zeros: 59946631.0 | params norm: 1109.783 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-24 09:55:03] iteration      220/   40000 | consumed samples:        14080 | elapsed time per iteration (ms): 1566.9 | throughput per GPU (TFLOP/s/GPU): 12.4 | learning rate: 1.320000E-04 | global batch size:    64 | lm loss: 6.749821E+00 | load_balancing_loss: 1.013973E+00 | loss scale: 1.0 | grad norm: 0.422 | num zeros: 60183509.0 | params norm: 1111.244 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

</details>