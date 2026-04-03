import os
import argparse
import glob
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from balm import (
    MoETrainer,
    BalmMoEConfig,
    BalmMoEForMaskedLM,
    BalmTokenizer,
)

# data paths - update these to point to your local copies of the training data
SHARDS_DIR = "./unpaired-data/unpaired-train-shards/"
UNPAIRED_EVAL = "./unpaired-data/unpaired-eval.parquet"


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--router", type=str, choices=["top-k", "top-p", "expert-choice"])
    p.add_argument("--expert_capacity", type=float)
    p.add_argument("--num_shared_experts", type=int)
    p.add_argument("--num_experts", type=int)
    p.add_argument("--expert_intermediate_size", type=int)
    return p.parse_args()


def main():
    args = parser()

    run_name = f"{args.router}_{args.expert_capacity}-ec_{args.num_shared_experts}-shared"

    # tokenizer
    tokenizer = BalmTokenizer()

    # dataset
    data_files = {
        "train": glob.glob(f"{SHARDS_DIR}*.parquet"),
        "eval": UNPAIRED_EVAL,
    }
    dataset = load_dataset("parquet", data_files=data_files)

    # tokenize
    # Note: if you change the tokenization, delete the .cache/ directory
    # manually to avoid reusing a stale cache
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["sequence"],
            padding="max_length",
            truncation=True,
            max_length=160,
        ),
        remove_columns="sequence_id",
        num_proc=128,
        cache_file_names={k: f"./.cache/{str(k)}.arrow" for k in dataset},
    )

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # wandb (optional)
    os.environ["WANDB_PROJECT"] = "BALM-MoE"
    os.environ["WANDB_RUN_GROUP"] = "45M"
    os.environ["WANDB_JOB_TYPE"] = str(args.router)

    config = BalmMoEConfig(
        hidden_size=480,
        intermediate_size=1920,
        num_hidden_layers=12,
        num_attention_heads=20,
        activation="swiglu",
        return_dict=True,
        # MoE params
        num_experts=args.num_experts,
        router_type=args.router,
        expert_capacity_type="multiplier",
        expert_capacity=args.expert_capacity,
        num_experts_per_tok=2,
        expert_activation="swiglu",
        num_initial_dense_layers=1,
        num_shared_experts=args.num_shared_experts,
        expert_intermediate_size=args.expert_intermediate_size,
    )
    model = BalmMoEForMaskedLM(config)

    training_args = TrainingArguments(
        seed=42,
        eval_strategy="steps",
        max_steps=250000,
        save_strategy="steps",
        save_steps=5000,
        logging_steps=50,
        eval_steps=5000,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        warmup_steps=15000,
        bf16=True,
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        run_name=run_name,
        output_dir=f"./checkpoints/{run_name}",
        overwrite_output_dir=True,
        logging_dir=f"./logs/{run_name}",
        logging_first_step=True,
        report_to="wandb",
    )

    trainer = MoETrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
    )

    trainer.train()
    trainer.save_model(f"./models/{run_name}")


if __name__ == "__main__":
    main()
