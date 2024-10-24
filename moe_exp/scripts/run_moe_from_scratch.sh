MASTER_ADDR="localhost"
MASTER_PORT=23456
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0

# TRAIN_DATA_PATH="none"
# VALID_DATA_PATH="none"
TRAIN_DATA_PATH="/mnt/chatbot30TB/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document"
VALID_DATA_PATH="/mnt/chatbot30TB/shseo/Megatron-LM/moe_exp/data/processed/llama3_fineweb_eval_text_document"

TOKENIZER_MODEL_PATH="/mnt/chatbot30TB/shseo/ckpt/llama3/meta-llama-3.1-8B"
TOKENIZER_TYPE="HuggingFaceTokenizer"
CHECKPOINT_PATH="/mnt/chatbot30TB/shseo/checkpoint/megatron/moe"
mkdir -p $CHECKPOINT_PATH

USE_MOE=true
NUM_EXPERTS=8
TOPK=2
MOE_TYPE="dmoe"

TP=1
PP=1
VPP="none"
EP=1

export WANDB_API_KEY="4a6b29eb77d6a110c7e361a8917abdccd3e0b8b1"
export WANDB_PROJECT="megatron"
export WANDB_NAME="moe_${USE_MOE}_num_experts_${NUM_EXPERTS}_topk_${TOPK}_type_${MOE_TYPE}_num_nodes_${NNODES}_gpus_per_node_${GPUS_PER_NODE}_TP_${TP}_PP_${PP}_VPP_${VPP}_EP_${EP}"

./moe_exp/scripts/run_moe.sh \
$MASTER_ADDR $MASTER_PORT $GPUS_PER_NODE $NNODES $NODE_RANK \
$TRAIN_DATA_PATH $VALID_DATA_PATH $TOKENIZER_MODEL_PATH $TOKENIZER_TYPE $CHECKPOINT_PATH \
$USE_MOE $NUM_EXPERTS $TOPK $MOE_TYPE \
$TP $PP $VPP $EP


