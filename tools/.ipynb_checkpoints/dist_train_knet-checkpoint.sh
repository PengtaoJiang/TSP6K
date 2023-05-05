CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

echo "$(dirname $0)"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
hfai python $(dirname "$0")/launch.py  \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} \
    --auto-resume \
    -- -n 1 -i ubuntu2004-cu113-ext --name knet_s3_upernet_r50-d8_80k \
    # ++ -ss 100 -ls 1 \


