# !/bin/bash
set -x


# Set default values for single-node training
nnodes=1                  # number of nodes
nproc_per_node=4          # number of processes per node
node_rank=0               # node rank
master_addr="localhost"   # address of the master node
master_port=29500         # port of the master node


torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py "$@"


#bash scripts/tokenizer/train_vq.sh --cloud-save-path . --data-path E:\.cache\ILSVRC___imagenet-1k\default\1.0.0\09dbb3153f1ac686bac1f40d24f307c383b383bc171f2df5d9e91c1ad57455b9 --image-size 256 --vq-model VQ-16

#E:\.cache\ILSVRC___imagenet-1k\default\1.0.0\09dbb3153f1ac686bac1f40d24f307c383b383bc171f2df5d9e91c1ad57455b9