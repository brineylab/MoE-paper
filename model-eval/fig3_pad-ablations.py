from ablm_eval import (
    InferenceConfig,
    RoutingConfig,
    evaluate_ablms,
    DatasetColumns,
)

# models
MODELS = {
    "standard": "./models/top-2_mask-aux-loss-False_mask-pad-probs-False/",
    "mask-pad-probs": "./models/top-2_mask-aux-loss-False_mask-pad-probs-True/",
    "mask-aux-loss": "./models/top-2_mask-aux-loss-True_mask-pad-probs-False/",
    "mask-aux-loss_mask-pad-probs": "./models/top-2_mask-aux-loss-True_mask-pad-probs-True/",
}

# data
UNPAIRED_DIR = "./unpaired-eval-data/"


def main():
    shared_output_dir = "./results/fig3/"

    configs = [
        InferenceConfig(
            dataset_name="heavy-100k",
            antibody_datatype="unpaired",
            data_path=f"{UNPAIRED_DIR}unpaired-test_heavy-100k-sampled_20251017_annotated.parquet",
            return_moe_losses=True,
            dataset_columns=DatasetColumns(chain_columns=["sequence"]),
            max_len=160,
        ),
        InferenceConfig(
            dataset_name="light-100k",
            antibody_datatype="unpaired",
            data_path=f"{UNPAIRED_DIR}unpaired-test_light-100k-sampled_20251017_annotated.parquet",
            return_moe_losses=True,
            dataset_columns=DatasetColumns(chain_columns=["sequence"]),
            max_len=160,
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
            max_len=160,
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
            max_len=160,
        ),
    ]

    evaluate_ablms(MODELS, configs, shared_output_dir, ignore_existing_files=False)


if __name__ == "__main__":
    main()
