from ablm_eval import (
    InferenceConfig,
    evaluate_ablms,
    DatasetColumns,
)

# models
MODELS = {
    # top-k
    "top-k_0.25-ec_0-shared": "./models/top-k_0.25-ec_0-shared/",
    "top-k_0.5-ec_0-shared": "./models/top-k_0.5-ec_0-shared/",
    "top-k_1.0-ec_0-shared": "./models/top-k_1.0-ec_0-shared/",
    "top-k_0.25-ec_1-shared": "./models/top-k_0.25-ec_1-shared/",
    "top-k_0.5-ec_1-shared": "./models/top-k_0.5-ec_1-shared/",
    "top-k_1.0-ec_1-shared": "./models/top-k_1.0-ec_1-shared/",
    # top-p
    "top-p_0.25-ec_0-shared": "./models/top-p_0.25-ec_0-shared/",
    "top-p_0.5-ec_0-shared": "./models/top-p_0.5-ec_0-shared/",
    "top-p_1.0-ec_0-shared": "./models/top-p_1.0-ec_0-shared/",
    "top-p_0.25-ec_1-shared": "./models/top-p_0.25-ec_1-shared/",
    "top-p_0.5-ec_1-shared": "./models/top-p_0.5-ec_1-shared/",
    "top-p_1.0-ec_1-shared": "./models/top-p_1.0-ec_1-shared/",
    # expert choice
    "expert-choice_0.25-ec_0-shared": "./models/expert-choice_0.25-ec_0-shared/",
    "expert-choice_0.5-ec_0-shared": "./models/expert-choice_0.5-ec_0-shared/",
    "expert-choice_1.0-ec_0-shared": "./models/expert-choice_1.0-ec_0-shared/",
    "expert-choice_0.25-ec_1-shared": "./models/expert-choice_0.25-ec_1-shared/",
    "expert-choice_0.5-ec_1-shared": "./models/expert-choice_0.5-ec_1-shared/",
    "expert-choice_1.0-ec_1-shared": "./models/expert-choice_1.0-ec_1-shared/",
}

# data
UNPAIRED_DIR = "./unpaired-eval-data/"


def main():
    shared_output_dir = "./results/fig1/"

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
    ]

    evaluate_ablms(MODELS, configs, shared_output_dir, ignore_existing_files=False)


if __name__ == "__main__":
    main()
