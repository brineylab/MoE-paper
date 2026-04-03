import os
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from balm import (
    BalmConfig,
    BalmForMaskedLM,
    BalmTokenizer,
    MixedDatasetCallback,
    process_mixed_dataset,
)

# data paths - update these to point to your local copies of the training data
PAIRED_DIR = "./paired-data/"
SHARDS_DIR = "./unpaired-data/unpaired-train-shards/"
UNPAIRED_EVAL = "./unpaired-data/unpaired-eval.parquet"


def main():
    run_name = "BALM-dense_mixed_710M"

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

    config = BalmConfig(
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=27,
        num_attention_heads=20,
        activation="swiglu",
        return_dict=True,
    )
    model = BalmForMaskedLM(config)

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

    trainer = Trainer(
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
