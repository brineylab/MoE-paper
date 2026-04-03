import os
import glob
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from balm import (
    BalmConfig,
    BalmForMaskedLM,
    BalmTokenizer,
)

# data paths - update these to point to your local copies of the training data
SHARDS_DIR = "./unpaired-data/unpaired-train-shards/"
UNPAIRED_EVAL = "./unpaired-data/unpaired-eval.parquet"


def main():
    run_name = "BALM-dense_45M_unpaired-only"

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
    os.environ["WANDB_JOB_TYPE"] = "dense"

    config = BalmConfig(
        hidden_size=480,
        intermediate_size=1920,
        num_hidden_layers=12,
        num_attention_heads=20,
        activation="swiglu",
        return_dict=True,
    )
    model = BalmForMaskedLM(config)

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

    trainer = Trainer(
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
