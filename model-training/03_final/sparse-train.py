import os
import argparse
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from balm import (
    MoETrainer,
    BalmMoEConfig,
    BalmMoEForMaskedLM,
    BalmTokenizer,
    MixedDatasetCallback,
    process_mixed_dataset,
)

# data paths - update these to point to your local copies of the training data
PAIRED_DIR = "./paired-data/"
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
        f"BALM-MoE_top-2_mixed_200M-act"
        f"_mask-aux-loss-{str(args.router_mask_aux_loss)}"
        f"_mask-pad-probs-{str(args.router_mask_pad_probs)}"
    )

    # tokenizer
    tokenizer = BalmTokenizer()

    # dataset
    data_files = {
        "paired_train": f"{PAIRED_DIR}paired-train_20250724.parquet",
        "unpaired_train": [
            os.path.join(SHARDS_DIR, f)
            for f in os.listdir(SHARDS_DIR)
            if f.endswith(".parquet")
        ],
        "paired_eval": f"{PAIRED_DIR}paired-eval_20250724.parquet",
        "unpaired_eval": UNPAIRED_EVAL,
    }
    train_dataset, eval_dataset = process_mixed_dataset(
        data_files=data_files,
        tokenizer=tokenizer,
        max_len=256,
        num_training_steps=500000,
        constant_prob=0.625,
        seed=42,
    )

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # wandb (optional)
    os.environ["WANDB_PROJECT"] = "BALM-MoE"
    os.environ["WANDB_RUN_GROUP"] = "large-scale_mixed"
    os.environ["WANDB_JOB_TYPE"] = "pre-training"

    config = BalmMoEConfig(
        hidden_size=640,
        intermediate_size=2560,
        num_hidden_layers=30,
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
        max_steps=500000,
        save_strategy="steps",
        save_steps=10000,
        logging_steps=50,
        eval_steps=10000,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        warmup_steps=30000,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[MixedDatasetCallback(train_dataset)],
    )

    trainer.train()
    trainer.save_model(f"./models/{run_name}")


if __name__ == "__main__":
    main()
