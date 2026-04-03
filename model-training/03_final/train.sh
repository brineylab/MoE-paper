#!/bin/bash

# error handling
error_handling() {
    echo "Error on line $1"
    exit 1
}
trap 'error_handling $LINENO' ERR

# allow ctrl-c to kill all background jobs
trap 'echo "Caught SIGINT, stopping all background jobs"; kill 0' SIGINT

# wandb
wandb login

### train
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router_mask_aux_loss true  --router_mask_pad_probs true
accelerate launch --config_file accelerate-config.yaml dense-200M-train.py
accelerate launch --config_file accelerate-config.yaml dense-700M-train.py
