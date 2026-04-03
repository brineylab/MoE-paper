#!/bin/bash

error_handling() {
    echo "Error on line $1"
    exit 1
}
trap 'error_handling $LINENO' ERR
trap 'echo "Caught SIGINT, stopping all background jobs"; kill 0' SIGINT

wandb login

### top-k router — no shared experts
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router "top-k" --num_experts 8 --expert_capacity 0.25 --num_shared_experts 0 --expert_intermediate_size 7680
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router "top-k" --num_experts 8 --expert_capacity 0.5  --num_shared_experts 0 --expert_intermediate_size 3840
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router "top-k" --num_experts 8 --expert_capacity 1.0  --num_shared_experts 0 --expert_intermediate_size 1920

### top-k router — 1 shared expert
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router "top-k" --num_experts 8 --expert_capacity 0.25 --num_shared_experts 1 --expert_intermediate_size 1920
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router "top-k" --num_experts 8 --expert_capacity 0.5  --num_shared_experts 1 --expert_intermediate_size 1440
accelerate launch --config_file accelerate-config.yaml sparse-train.py --router "top-k" --num_experts 8 --expert_capacity 1.0  --num_shared_experts 1 --expert_intermediate_size 960
