#!/bin/bash

source activate pyg

DATA=$1
MODEL=$2
SEED=$3
UN=original

export WANDB_MODE=offline
export WANDB_PROJECT=zitniklab-gnn-unlearning

export WANDB_NAME="$UN"_"$DATA"_"$MODEL"_"$SEED"
export WANDB_RUN_NAME="$UN"_"$DATA"_"$MODEL"_"$SEED"
export WANDB_RUN_ID="$UN"_"$DATA"_"$MODEL"_"$SEED"

python train_gnn.py --lr 1e-3 \
                        --epochs 1500 \
                        --dataset "$DATA" \
                        --random_seed "$SEED" \
                        --unlearning_model "$UN" \
                        --gnn "$MODEL"