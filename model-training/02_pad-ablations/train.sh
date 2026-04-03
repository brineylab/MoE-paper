#!/bin/bash

error_handling() {
    echo "Error on line $1"
    exit 1
}
trap 'error_handling $LINENO' ERR
trap 'echo "Caught SIGINT, stopping all background jobs"; kill 0' SIGINT

wandb login

### router_mask_aux_loss=false
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router_mask_aux_loss false --router_mask_pad_probs false
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router_mask_aux_loss false --router_mask_pad_probs true

### router_mask_aux_loss=true
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router_mask_aux_loss true  --router_mask_pad_probs false
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router_mask_aux_loss true  --router_mask_pad_probs true
