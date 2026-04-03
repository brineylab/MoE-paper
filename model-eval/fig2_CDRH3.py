from ablm_eval import (
    PerPositionConfig,
    RoutingConfig,
    evaluate_ablms,
    DatasetColumns,
)

# models
MODELS = {
    "top-k_1.0-ec_0-shared": "./models/top-k_1.0-ec_0-shared/",
    "top-p_1.0-ec_0-shared": "./models/top-p_1.0-ec_0-shared/",
    "expert-choice_1.0-ec_0-shared": "./models/expert-choice_1.0-ec_0-shared/",
}

# data
UNPAIRED_DIR = "./unpaired-eval-data/"


def main():
    shared_output_dir = "./results/fig2/"

    configs = [
        PerPositionConfig(
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
        PerPositionConfig(
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
