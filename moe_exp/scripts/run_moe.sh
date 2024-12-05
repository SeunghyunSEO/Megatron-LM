#!/bin/bash

# Runs Mixtral 8x7B model on 32 H100/A100 GPUs
# The Dropless MoE suffers from an imbalanced token distribution at the early stage of training (the first few hundred iterations), which may lead to poor performance and out-of-memory (OOM) issues.
# To check the performance of a Dropless MoE model, we should run the model for at least 500 iterations or resume from trained checkpoints.

export CUDA_DEVICE_MAX_CONNECTIONS=1

## hardware setup
MASTER_ADDR=${1:-"localhost"}
MASTER_PORT=${2:-23456}
GPUS_PER_NODE=${3:-8}
NNODES=${4:-2}
NODE_RANK=${5:-0}

TRAIN_DATA_PATH=${6:-"none"}
VALID_DATA_PATH=${7:-"none"}
TOKENIZER_MODEL_PATH=${8:-"none"}
TOKENIZER_TYPE=${9:-"none"}
CHECKPOINT_PATH=${10:-"none"}

USE_MOE=${11:-false}
NUM_EXPERTS=${12:-16}
TOPK=${13:-2}
MOE_TYPE=${14:-"dmoe"}

TP=${15:-2}
PP=${16:-2}
VPP=${17:-"none"}
EP=${18:-2}

LR=${19:-0.0006}
MIN_LR=${20:-0.00006}

MBSZ=${21:-1}
GBSZ=${22:-16}

TRAIN_ITERS=${23:-50000}
WARMUP_ITERS=${24:-2000}
DECAY_ITERS=${25:-48000}

LOG_INTERVAL=${26:-20}
SAVE_INTERVAL=${27:-1000000}
EVAL_INTERVAL=${28:-1000}
EVAL_ITERS=${29:-10}

UES_FSDP2=${30:-false}

WANDB_API_KEY=${oc.env:WANDB_API_KEY}
WANDB_PROJECT=${oc.env:WANDB_PROJECT}
WANDB_NAME=${oc.env:WANDB_NAME}


## dist training configs
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)
echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS[@]}"
echo ""

## model configs
MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 2048
    --position-embedding-type rope
    --rotary-base 100000
    # --num-layers 32
    --num-layers 24
    --hidden-size 512
    --num-attention-heads 4
    --group-query-attention
    --num-query-groups 1
    # --num-query-groups 2 # for testing TP
    --ffn-hidden-size 1792
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.02
    --untie-embeddings-and-output-weights
    # --no-masked-softmax-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
)
echo "MODEL_ARGS: ${MODEL_ARGS[@]}"
echo ""

## moe model configs
if [ "$USE_MOE" = true ]; then
    MOE_ARGS=(
        --num-experts "${NUM_EXPERTS}"
        --moe-router-topk "${TOPK}"
        --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
        --moe-aux-loss-coeff 1e-2
        --moe-token-dispatcher-type alltoall
        --moe-per-layer-logging
    )
    if [ "$TOPK" = "1" ]; then
        MOE_ARGS+=(--moe-router-pre-softmax)
    fi
    if [ "$MOE_TYPE" = "moe" ]; then
        MOE_ARGS+=(
            --moe-expert-capacity-factor 1.0
            --moe-pad-expert-input-to-capacity # Optional
        )
    elif [ "$MOE_TYPE" = "dmoe" ]; then
        MOE_ARGS+=(--moe-grouped-gemm)
    elif [ "$MOE_TYPE" = "mb_dmoe" ]; then
        MOE_ARGS+=(--moe-use-megablocks-dmoe)
    fi
else
    MOE_ARGS=()
fi
echo "MOE_ARGS: ${MOE_ARGS[@]}"
echo ""

## distributed training configs
if [ "$USE_FSDP2" = true ]; then
    MODEL_PARALLEL_ARGS=(
        --use-torch-fsdp2 # https://github.com/NVIDIA/Megatron-LM/commit/e1993fa6f70763523a84432ab1f5eb42e77ccf2a#diff-417e1412ae83b0a3bf40f2e95f78feb929a3b536a90bfa2b537299a563dbc5f4
    )
else
    MODEL_PARALLEL_ARGS=(
        --use-distributed-optimizer
    )
fi
MODEL_PARALLEL_ARGS+=(
    --tensor-model-parallel-size ${TP}
    --no-async-tensor-model-parallel-allreduce
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
)
if [ "$USE_FSDP2" = false ]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi
if [ $VPP != "none" ]; then
    MODEL_PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage ${VPP})
fi
echo "MODEL_PARALLEL_ARGS: ${MODEL_PARALLEL_ARGS[@]}"

## data args
DATA_ARGS=(
    --tokenizer-type "${TOKENIZER_TYPE}"
    --tokenizer-model "${TOKENIZER_MODEL_PATH}"
)
if [ "${TRAIN_DATA_PATH}" == "none" ]; then
    DATA_ARGS+=(--mock-data)
else
    if [ "${VALID_DATA_PATH}" == "none" ]; then
        DATA_ARGS+=(
            --data-path ${TRAIN_DATA_PATH} 
            --split 99990,8,2
        )
    else
        DATA_ARGS+=(
            --train-data-path ${TRAIN_DATA_PATH} 
            --valid-data-path ${VALID_DATA_PATH} 
        )
    fi
fi
echo "DATA_ARGS: ${DATA_ARGS[@]}"
echo ""

## hparams
TRAINING_ARGS=(
    --micro-batch-size ${MBSZ}
    --global-batch-size ${GBSZ}
    --train-iters ${TRAIN_ITERS}
    --lr-decay-iters ${DECAY_ITERS}
    --lr-warmup-iters ${WARMUP_ITERS}
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --use-flash-attn
    # --fp32-residual-connection
    # --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --overlap-grad-reduce
    --overlap-param-gather
    --no-gradient-accumulation-fusion # apex version issue
)
echo "TRAINING_ARGS: ${TRAINING_ARGS[@]}"
echo ""

## saving arguments
SAVING_ARGS=(
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --save-interval ${SAVE_INTERVAL}
    # --use-dist-ckpt
    # --auto-detect-ckpt-format
    # --dist-ckpt-format zarr
    # --ckpt-fully-parallel-save
    # --use-checkpoint-opt_param-scheduler
)
echo "SAVING_ARGS: ${SAVING_ARGS[@]}"
echo ""

## logger arguments
LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --log-throughput
    --log-params-norm
    --log-num-zeros-in-grad
    --log-progress
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --tensorboard-log-interval ${LOG_INTERVAL}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-world-size-to-tensorboard
    # --log-batch-size-to-tensorboard
    --no-load-optim
    --no-load-rng
)
if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"megatron"}
        --wandb-exp-name ${WANDB_NAME:-"moe_from_scratch"} 
    )
fi
echo "LOGGING_ARGS: ${LOGGING_ARGS[@]}"
echo ""

## run
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${SAVING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
