from ablm_eval import (
    InferenceConfig,
    RoutingConfig,
    ClassificationConfig,
    evaluate_ablms,
    DatasetColumns,
)
from transformers import TrainingArguments

# models
MODELS = {
    "dense-200M_mixed": "./models/BALM-dense_mixed_200M/",
    "Top-2_200M-act_mixed": "./models/BALM-MoE_top-2_mixed_200M-act_mask-aux-loss-True_mask-pad-probs-True/",
    "dense-710M_mixed": "./models/BALM-dense_mixed_710M/",
}

# data
PAIRED_DIR = "./paired-eval-data/"
UNPAIRED_DIR = "./unpaired-eval-data/"
CLASS_DIR = "./specificity-classification-data/"


def main():
    shared_output_dir = "./results/fig4/"

    configs = [
        # inference
        InferenceConfig(
            dataset_name="paired",
            antibody_datatype="paired",
            data_path=f"{PAIRED_DIR}paired-test_20250724.parquet",
            return_moe_losses=True,
            dataset_columns=DatasetColumns(chain_columns=["sequence"]),
            max_len=256,
        ),
        InferenceConfig(
            dataset_name="heavy-100k",
            antibody_datatype="unpaired",
            data_path=f"{UNPAIRED_DIR}unpaired-test_heavy-100k-sampled_20251017_annotated.parquet",
            return_moe_losses=True,
            dataset_columns=DatasetColumns(chain_columns=["sequence"]),
            max_len=256,
        ),
        InferenceConfig(
            dataset_name="light-100k",
            antibody_datatype="unpaired",
            data_path=f"{UNPAIRED_DIR}unpaired-test_light-100k-sampled_20251017_annotated.parquet",
            return_moe_losses=True,
            dataset_columns=DatasetColumns(chain_columns=["sequence"]),
            max_len=256,
        ),
        # routing
        RoutingConfig(
            dataset_name="paired",
            antibody_datatype="paired",
            data_path=f"{PAIRED_DIR}paired-test_5k-sampled_20251215.parquet",
            dataset_columns=DatasetColumns(
                cdr_columns=["cdr_mask_aa_heavy", "cdr_mask_aa_light"]
            ),
            keep_columns=[
                "nongermline_mask_aa_heavy",
                "gene_segment_mask_aa_heavy",
                "nongermline_mask_aa_light",
                "gene_segment_mask_aa_light",
            ],
            max_len=256,
        ),
        RoutingConfig(
            dataset_name="heavy",
            antibody_datatype="unpaired",
            data_path=f"{UNPAIRED_DIR}unpaired-test_heavy_5k-sampled_20251017.parquet",
            dataset_columns=DatasetColumns(
                chain_columns=["sequence"],
                cdr_columns=["cdr_mask_aa"],
            ),
            keep_columns=["nongermline_mask_aa", "gene_segment_mask_aa"],
            max_len=256,
        ),
        RoutingConfig(
            dataset_name="light",
            antibody_datatype="unpaired",
            data_path=f"{UNPAIRED_DIR}unpaired-test_light_5k-sampled_20251017.parquet",
            dataset_columns=DatasetColumns(
                chain_columns=["sequence"],
                cdr_columns=["cdr_mask_aa"],
            ),
            keep_columns=["nongermline_mask_aa", "gene_segment_mask_aa"],
            max_len=256,
        ),
        # specificity classification
        ClassificationConfig(
            dataset_name="HD-CoV",
            antibody_datatype="paired",
            launcher="accelerate",
            num_folds=5,
            num_classes=2,
            data_path={
                i: {
                    "train": f"{CLASS_DIR}/HD-CoV/TTE/hd-0_cov-1_train{i}.csv",
                    "test": f"{CLASS_DIR}/HD-CoV/TTE/hd-0_cov-1_test{i}.csv",
                }
                for i in range(5)
            },
            max_len=256,
            training_args=TrainingArguments(
                run_name="",
                seed=42,
                bf16=True,
                learning_rate=5e-5,
                per_device_train_batch_size=32,
                num_train_epochs=5,
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
                eval_strategy="steps",
                eval_steps=250,
                per_device_eval_batch_size=32,
                eval_accumulation_steps=50,
                logging_steps=50,
                save_strategy="no",
                output_dir="",
                report_to="none",
                logging_first_step=True,
            ),
        ),
        ClassificationConfig(
            dataset_name="HD-CoV-Flu",
            antibody_datatype="paired",
            launcher="accelerate",
            num_folds=5,
            num_classes=3,
            data_path={
                i: {
                    "train": f"{CLASS_DIR}/HD-CoV-Flu/TTE/hd-0_cov-1_flu-2_train{i}.csv",
                    "test": f"{CLASS_DIR}/HD-CoV-Flu/TTE/hd-0_cov-1_flu-2_test{i}.csv",
                }
                for i in range(5)
            },
            max_len=256,
            training_args=TrainingArguments(
                run_name="",
                seed=42,
                bf16=True,
                learning_rate=5e-5,
                per_device_train_batch_size=8,
                num_train_epochs=5,
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
                eval_strategy="steps",
                eval_steps=250,
                per_device_eval_batch_size=32,
                eval_accumulation_steps=50,
                logging_steps=50,
                save_strategy="no",
                output_dir="",
                report_to="none",
                logging_first_step=True,
            ),
        ),
    ]

    evaluate_ablms(MODELS, configs, shared_output_dir, ignore_existing_files=False)


if __name__ == "__main__":
    main()
