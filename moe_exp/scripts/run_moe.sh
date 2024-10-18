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

CHECKPOINT_PATH=${6:-"none"}
DATA_PATH=${7:-"none"}
TOKENIZER_MODEL_PATH=${8:-"none"}
TOKENIZER_TYPE=${9:-"none"}

NUM_EXPERTS=${10:-16}
TOPK=${11:-2}

TP=${12:-2}
PP=${13:-2}
EP=${14:-2}

USE_MB_DMOE=${15:-false}

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

## model/tokenizer/data path
CHECKPOINT_PATH=${CHECKPOINT_PATH}
DATA_PATH=${DATA_PATH}
TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH}
TOKENIZER_TYPE=${TOKENIZER_TYPE}

## model configs
MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 256
    --max-position-embeddings 32768
    --num-layers 4
    --hidden-size 256
    --ffn-hidden-size 1024
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --num-attention-heads 4
    --group-query-attention
    --num-query-groups 2
    --no-masked-softmax-fusion
    --no-position-embedding
)
echo "MODEL_ARGS: ${MODEL_ARGS[@]}"

## moe model configs
MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --expert-model-parallel-size ${EP}
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk ${TOPK}
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
)
if [ $USE_MB_DMOE = true ]; then
    MOE_ARGS+=("--moe-use-megablocks-dmoe")
fi
echo "MOE_ARGS: ${MOE_ARGS[@]}"

## distributed training configs
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    # --num-layers-per-virtual-pipeline-stage # ignore interleaving 
    --sequence-parallel
    --use-distributed-optimizer
)
echo "MODEL_PARALLEL_ARGS: ${MODEL_PARALLEL_ARGS[@]}"

## data args
DATA_ARGS=(
    --tokenizer-type "${TOKENIZER_TYPE}"
    --tokenizer-model "${TOKENIZER_MODEL_PATH}"
)
if [ "${DATA_PATH}" == "none" ]; then
    DATA_ARGS+=("--mock-data")
else
    DATA_ARGS+=("--data-path" "${DATA_PATH}" "--split" "99990,8,2")
fi
echo "DATA_ARGS: ${DATA_ARGS[@]}"

## hparams
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
    --no-gradient-accumulation-fusion # apex version issue
)
echo "TRAINING_ARGS: ${TRAINING_ARGS[@]}"

## logger arguments
LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)
if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
    )
fi
echo "LOGGING_ARGS: ${LOGGING_ARGS[@]}"

## run
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
