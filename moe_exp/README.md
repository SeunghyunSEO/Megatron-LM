
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

```
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

## pdb tracer in dist setting

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


## run scripts for torch native parallelism

```bash
cd /path/to/dir/Megatron-LM/moe_exp
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
cd /path/to/dir/Megatron-LM/moe_exp
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
cd /path/to/dir/Megatron-LM/moe_exp
export LOCAL_RANK=0 &&\
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23458 &&\
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_dmoe.py
```


## run scripts for megatron integrated dmoe

```bash
cd /path/to/dir/Megatron-LM/
bash ./moe_exp/scripts/run_moe_from_scratch.sh
```

<details>

- node=1 / ngpu=2 / DP:TP:PP:EP=2:1:1:1 

```python
MODEL_ARGS: --disable-bias-linear --seq-length 256 --max-position-embeddings 32768 --num-layers 4 --hidden-size 256 --ffn-hidden-size 1024 --init-method-std 0.01 --attention-dropout 0.0 --hidden-dropout 0.0 --normaliza
tion RMSNorm --position-embedding-type rope --swiglu --untie-embeddings-and-output-weights --num-attention-heads 4 --group-query-attention --num-query-groups 2 --no-masked-softmax-fusion --no-position-embedding        
MOE_ARGS: --num-experts 8 --expert-model-parallel-size 1 --moe-router-load-balancing-type aux_loss --moe-router-topk 2 --moe-aux-loss-coeff 1e-2 --moe-grouped-gemm                                                       
MODEL_PARALLEL_ARGS: --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --sequence-parallel --use-distributed-optimizer                                                                                      
DATA_ARGS: --tokenizer-type HuggingFaceTokenizer --tokenizer-model /workspace/ckpt/llama3/meta-llama-3.1-8B --mock-data                                                                                       
TRAINING_ARGS: --micro-batch-size 1 --global-batch-size 128 --lr 1e-4 --train-iters 500000 --lr-decay-iters 320000 --lr-decay-style cosine --min-lr 1.0e-5 --weight-decay 0.1 --lr-warmup-iters 500 --clip-grad 1.0 --bf16 --overlap-grad-reduce --overlap-param-gather --no-gradient-accumulation-fusion
```

```python
Params for bucket 1 (41421568 elements):
        module.decoder.layers.3.mlp.experts.weight1
        module.decoder.layers.3.pre_mlp_layernorm.weight
        module.decoder.layers.3.self_attention.linear_qkv.weight
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.mlp.experts.weight2
        module.output_layer.weight
        module.decoder.final_layernorm.weight
        module.decoder.layers.3.mlp.experts.weight2
        module.decoder.layers.3.mlp.router.weight
        module.decoder.layers.3.self_attention.linear_proj.weight
Params for bucket 2 (50208256 elements):
        module.decoder.layers.2.mlp.experts.weight1
        module.decoder.layers.2.mlp.router.weight
        module.decoder.layers.2.pre_mlp_layernorm.weight
        module.decoder.layers.1.pre_mlp_layernorm.weight
        module.decoder.layers.1.self_attention.linear_proj.weight
        module.decoder.layers.1.mlp.experts.weight1
        module.decoder.layers.0.mlp.router.weight
        module.decoder.layers.2.self_attention.linear_proj.weight
        module.decoder.layers.0.mlp.experts.weight2
        module.decoder.layers.0.self_attention.linear_proj.weight
        module.decoder.layers.1.mlp.router.weight
        module.embedding.word_embeddings.weight
        module.decoder.layers.1.mlp.experts.weight2
        module.decoder.layers.0.mlp.experts.weight1
        module.decoder.layers.0.pre_mlp_layernorm.weight
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.0.self_attention.linear_qkv.weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.self_attention.linear_qkv.weight
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight

INFO:megatron.core.optimizer:Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=0.0001, min_lr=1e-05, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.1, fp16=False, bf16=True, params_dtype=t
orch.bfloat16, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=Tru
e, overlap_param_gather_with_optimizer_step=False, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=True, timers=<megatron.core.timers.Timers object at 0x7f819418b2e0>, config_logger_dir='')
INFO:megatron.core.optimizer_param_scheduler:> learning rate decay style: cosine
```

```python
 [2024-10-16 04:23:00] iteration        2/  500000 | consumed samples:          256 | elapsed time per iteration (ms): 1811.8 | learning rate: 4.000000E-07 | global batch size:   128 | lm loss: 1.177651E+01 | load_balancing_loss: 1.008666E+00 | loss scale: 1.0 | grad norm: 0.493 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-16 04:23:02] iteration        3/  500000 | consumed samples:          384 | elapsed time per iteration (ms): 1782.9 | learning rate: 6.000000E-07 | global batch size:   128 | lm loss: 1.177523E+01 | load_balancing_loss: 1.009316E+00 | loss scale: 1.0 | grad norm: 0.476 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-16 04:23:04] iteration        4/  500000 | consumed samples:          512 | elapsed time per iteration (ms): 1778.7 | learning rate: 8.000000E-07 | global batch size:   128 | lm loss: 1.177628E+01 | load_balancing_loss: 1.009021E+00 | loss scale: 1.0 | grad norm: 0.454 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-16 04:23:06] iteration        5/  500000 | consumed samples:          640 | elapsed time per iteration (ms): 1781.9 | learning rate: 1.000000E-06 | global batch size:   128 | lm loss: 1.177547E+01 | load_balancing_loss: 1.008862E+00 | loss scale: 1.0 | grad norm: 0.463 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-16 04:23:08] iteration        6/  500000 | consumed samples:          768 | elapsed time per iteration (ms): 1774.3 | learning rate: 1.200000E-06 | global batch size:   128 | lm loss: 1.177632E+01 | load_balancing_loss: 1.009468E+00 | loss scale: 1.0 | grad norm: 0.464 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

- node=1 / ngpu=2 / DP:TP:PP:EP=1:1:1:2

```python
MODEL_ARGS: --disable-bias-linear --seq-length 256 --max-position-embeddings 32768 --num-layers 4 --hidden-size 256 --ffn-hidden-size 1024 --init-method-std 0.01 --attention-dropout 0.0 --hidden-dropout 0.0 --normaliza
tion RMSNorm --position-embedding-type rope --swiglu --untie-embeddings-and-output-weights --num-attention-heads 4 --group-query-attention --num-query-groups 2 --no-masked-softmax-fusion --no-position-embedding
MOE_ARGS: --num-experts 8 --expert-model-parallel-size 2 --moe-router-load-balancing-type aux_loss --moe-router-topk 2 --moe-aux-loss-coeff 1e-2 --moe-grouped-gemm
MODEL_PARALLEL_ARGS: --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --sequence-parallel --use-distributed-optimizer
DATA_ARGS: --tokenizer-type HuggingFaceTokenizer --tokenizer-model /workspace/ckpt/llama3/meta-llama-3.1-8B --mock-data
TRAINING_ARGS: --micro-batch-size 1 --global-batch-size 128 --lr 1e-4 --train-iters 500000 --lr-decay-iters 320000 --lr-decay-style cosine --min-lr 1.0e-5 --weight-decay 0.1 --lr-warmup-iters 500 --clip-grad 1.0 --bf16 --overlap-grad-reduce --overlap-param-gather --no-gradient-accumulation-fusion
```

```python
WARNING:megatron.core.tensor_parallel.random:CPU RNG state changed within GPU RNG context
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 79046912
INFO:megatron.core.distributed.distributed_data_parallel:Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=True, overlap_param_gather=True, align
_param_gather=False, use_distributed_optimizer=True, check_for_nan_in_grad=True, bucket_size=40000000, average_in_collective=False, fp8_param_gather=False)
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 1
Params for bucket 1 (66464000 elements):
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.embedding.word_embeddings.weight
        module.decoder.layers.3.self_attention.linear_qkv.weight
        module.decoder.layers.1.pre_mlp_layernorm.weight
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.layers.0.self_attention.linear_qkv.weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.3.pre_mlp_layernorm.weight
        module.decoder.final_layernorm.weight
        module.decoder.layers.3.self_attention.linear_proj.weight
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.mlp.router.weight
        module.decoder.layers.1.mlp.router.weight
        module.decoder.layers.0.self_attention.linear_proj.weight
        module.decoder.layers.2.pre_mlp_layernorm.weight
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.self_attention.linear_proj.weight
        module.decoder.layers.0.mlp.router.weight
        module.output_layer.weight
        module.decoder.layers.2.self_attention.linear_qkv.weight
        module.decoder.layers.1.self_attention.linear_proj.weight
        module.decoder.layers.3.mlp.router.weight
        module.decoder.layers.0.pre_mlp_layernorm.weight
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 1
Params for bucket 1 (12582912 elements):
        module.decoder.layers.1.mlp.experts.weight2
        module.decoder.layers.3.mlp.experts.weight1
        module.decoder.layers.0.mlp.experts.weight2
        module.decoder.layers.3.mlp.experts.weight2
        module.decoder.layers.2.mlp.experts.weight1
        module.decoder.layers.2.mlp.experts.weight2
        module.decoder.layers.1.mlp.experts.weight1
        module.decoder.layers.0.mlp.experts.weight1
INFO:megatron.core.optimizer:Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=0.0001, min_lr=1e-05, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.1, fp16=False, bf16=True, params_dtype=t
orch.bfloat16, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=Tru
e, overlap_param_gather_with_optimizer_step=False, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=True, timers=<megatron.core.timers.Timers object at 0x7f341041e530>, config_logger_dir='')
INFO:megatron.core.optimizer_param_scheduler:> learning rate decay style: cosine
WARNING: could not find the metadata file /workspace/checkpoint/megatron/moe/latest_checkpointed_iteration.txt
    will not load any checkpoints and will start from random
/workspace/.local/lib/python3.10/site-packages/torch/distributed/c10d_logger.py:79: FutureWarning: `torch.distributed._all_gather_base` is a private function and will be deprecated. Please use `torch.distributed.all_ga
ther_into_tensor` instead.
  return func(*args, **kwargs)
/workspace/.local/lib/python3.10/site-packages/torch/distributed/c10d_logger.py:79: FutureWarning: `torch.distributed._all_gather_base` is a private function and will be deprecated. Please use `torch.distributed.all_ga
ther_into_tensor` instead.
  return func(*args, **kwargs)
```

```python
 [2024-10-16 06:21:10] iteration        1/  500000 | consumed samples:          128 | elapsed time per iteration (ms): 14312.0 | learning rate: 2.000000E-07 | global batch size:   128 | lm loss: 1.177436E+01 | load_bal
ancing_loss: 1.010554E+00 | loss scale: 1.0 | grad norm: 0.479 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
Number of parameters in transformer layers in billions:  0.03
Number of parameters in embedding layers in billions: 0.07
Total number of parameters in billions: 0.09
Number of parameters in most loaded shard in billions: 0.0916
Theoretical memory footprints: weight and optimizer=1048.55 MB
[Rank 0] (after 1 iterations) memory (MB) | allocated: 999.52783203125 | max allocated: 999.5595703125 | reserved: 1126.0 | max reserved: 1126.0
 [2024-10-16 06:21:12] iteration        2/  500000 | consumed samples:          256 | elapsed time per iteration (ms): 2034.8 | learning rate: 4.000000E-07 | global batch size:   128 | lm loss: 1.177444E+01 | load_bala
ncing_loss: 1.009700E+00 | loss scale: 1.0 | grad norm: 0.489 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-16 06:21:14] iteration        3/  500000 | consumed samples:          384 | elapsed time per iteration (ms): 2049.3 | learning rate: 6.000000E-07 | global batch size:   128 | lm loss: 1.177427E+01 | load_bala
ncing_loss: 1.010454E+00 | loss scale: 1.0 | grad norm: 0.476 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-16 06:21:16] iteration        4/  500000 | consumed samples:          512 | elapsed time per iteration (ms): 2037.0 | learning rate: 8.000000E-07 | global batch size:   128 | lm loss: 1.177555E+01 | load_bala
ncing_loss: 1.010483E+00 | loss scale: 1.0 | grad norm: 0.455 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-10-16 06:21:18] iteration        5/  500000 | consumed samples:          640 | elapsed time per iteration (ms): 2041.6 | learning rate: 1.000000E-06 | global batch size:   128 | lm loss: 1.177305E+01 | load_bala
ncing_loss: 1.009435E+00 | loss scale: 1.0 | grad norm: 0.463 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
^CW1016 06:21:18.872000 140016382740288 torch/distributed/elastic/agent/server/api.py:688] Received Signals.SIGINT death signal, shutting down workers 
```

- dmoe prototype

```python
Params for bucket 1 (40572160 elements):
        module.decoder.final_layernorm.weight
        module.decoder.layers.3.mlp.dmoe.router.layer.weight
        module.decoder.layers.1.mlp.dmoe.experts.mlp.v1
        module.output_layer.weight
        module.decoder.layers.3.pre_mlp_layernorm.weight
        module.decoder.layers.2.mlp.dmoe.experts.mlp.w1
        module.decoder.layers.2.mlp.dmoe.router.layer.weight
        module.decoder.layers.2.self_attention.linear_qkv.weight
        module.decoder.layers.3.mlp.dmoe.experts.mlp.w2
        module.decoder.layers.3.self_attention.linear_proj.weight
        module.decoder.layers.2.self_attention.linear_proj.weight
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.mlp.dmoe.experts.mlp.v1
        module.decoder.layers.3.mlp.dmoe.experts.mlp.v1
        module.decoder.layers.3.mlp.dmoe.experts.mlp.w1
        module.decoder.layers.3.self_attention.linear_qkv.weight
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.pre_mlp_layernorm.weight
        module.decoder.layers.2.mlp.dmoe.experts.mlp.w2
Params for bucket 2 (38474752 elements):
        module.decoder.layers.1.mlp.dmoe.experts.mlp.w2
        module.decoder.layers.1.pre_mlp_layernorm.weight
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.1.self_attention.linear_proj.weight
        module.decoder.layers.0.mlp.dmoe.experts.mlp.v1
        module.decoder.layers.0.mlp.dmoe.experts.mlp.w2
        module.decoder.layers.0.mlp.dmoe.experts.mlp.w1
        module.decoder.layers.0.mlp.dmoe.router.layer.weight
        module.decoder.layers.0.pre_mlp_layernorm.weight
        module.decoder.layers.1.mlp.dmoe.router.layer.weight
        module.decoder.layers.1.mlp.dmoe.experts.mlp.w1
        module.decoder.layers.0.self_attention.linear_proj.weight
        module.decoder.layers.0.self_attention.linear_qkv.weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.embedding.word_embeddings.weight
```

</details>