#!/bin/bash

source activate pyg

DATA=$1
MODEL=$2
UN=$3
DF=$4
DF_SIZE=$5
SEED=$6

export WANDB_MODE=offline
export WANDB_PROJECT=zitniklab-gnn-unlearning


export WANDB_NAME="$UN"_"$DATA"_"$MODEL"_"$DF"_"$DF_SIZE"_"$SEED"
export WANDB_RUN_NAME="$UN"_"$DATA"_"$MODEL"_"$DF"_"$DF_SIZE"_"$SEED"
export WANDB_RUN_ID="$UN"_"$DATA"_"$MODEL"_"$DF"_"$DF_SIZE"_"$SEED"

python delete_gnn.py --lr 1e-3 \
                        --epochs 1500 \
                        --dataset "$DATA" \
                        --random_seed "$SEED" \
                        --unlearning_model "$UN" \
                        --gnn "$MODEL" \
                        --df "$DF" \
                        --df_size "$DF_SIZE"