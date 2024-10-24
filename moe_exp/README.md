
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

```bash
# cd /path/to/dir/Megatron-LM/
bash ./moe_exp/scripts/run_moe_from_scratch.sh
```

- node=1 / ngpu=2 / DP:TP:(PP:VPP):EP=2:1:(1:1):1 

<details>

```
DISTRIBUTED_ARGS: --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 23456
MODEL_ARGS: --disable-bias-linear --seq-length 256 --max-position-embeddings 2048 --num-layers 4 --hidden-size 256 --ffn-hidden-size 1024 --init-method-std 0.01 --attention-dropout 0.0 --hidden-dropout 0.0 --normalization RMSNorm --position-embedding-type rope --swiglu --untie-embeddings-and-output-weights --num-attention-heads 4 --group-query-attention --num-query-groups 2 --no-masked-softmax-fusion --no-position-embedding
MOE_ARGS: --num-experts 8 --moe-router-topk 2 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2 --moe-token-dispatcher-type alltoall --moe-grouped-gemm
MODEL_PARALLEL_ARGS: --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 2 --sequence-parallel --use-distributed-optimizer
DATA_ARGS: --tokenizer-type HuggingFaceTokenizer --tokenizer-model /workspace/ckpt/llama3/meta-llama-3.1-8B --mock-data
TRAINING_ARGS: --micro-batch-size 1 --global-batch-size 128 --lr 1e-4 --train-iters 500000 --lr-decay-iters 320000 --lr-decay-style cosine --min-lr 1.0e-5 --weight-decay 0.1 --lr-warmup-iters 500 --clip-grad 1.0 --bf16 --overlap-grad-reduce --overlap-param-gather --no-gradient-accumulation-fusion
LOGGING_ARGS: --save /workspace/checkpoint/megatron/moe --load /workspace/checkpoint/megatron/moe --log-interval 1 --log-throughput --log-params-norm --log-num-zeros-in-grad --log-progress --save-interval 10000 --eval-interval 1000 --eval-iters 10 --tensorboard-dir /workspace/checkpoint/megatron/moe/tensorboard --tensorboard-log-interval 1 --log-timers-to-tensorboard --log-batch-size-to-tensorboard --log-validation-ppl-to-tensorboard --log-memory-to-tensorboard --log-world-size-to-tensorboard --no-load-optim --no-load-rng
```

</details>