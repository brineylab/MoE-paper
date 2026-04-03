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


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("yes", "y", "true", "t", "1"):
        return True
    if s in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--router_mask_aux_loss",
        type=str2bool,
        default=False,
        help="Boolean. Use true/false. Default: false",
        metavar="{true,false}",
    )
    p.add_argument(
        "--router_mask_pad_probs",
        type=str2bool,
        default=False,
        help="Boolean. Use true/false. Default: false",
        metavar="{true,false}",
    )
    return p.parse_args()


def main():
    args = parser()

    run_name = (
        f"top-2"
        f"_mask-aux-loss-{str(args.router_mask_aux_loss)}"
        f"_mask-pad-probs-{str(args.router_mask_pad_probs)}"
    )

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
    os.environ["WANDB_RUN_GROUP"] = "pad-token-ablations"
    os.environ["WANDB_JOB_TYPE"] = "pre-training"

    config = BalmMoEConfig(
        hidden_size=480,
        intermediate_size=1920,
        num_hidden_layers=12,
        num_attention_heads=20,
        activation="swiglu",
        return_dict=True,
        # MoE params
        num_experts=8,
        router_type="top-k",
        expert_capacity_type="multiplier",
        expert_capacity=1.0,
        num_experts_per_tok=2,
        expert_activation="swiglu",
        num_initial_dense_layers=1,
        num_shared_experts=0,
        router_mask_aux_loss=args.router_mask_aux_loss,
        router_mask_pad_probs=args.router_mask_pad_probs,
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
