import argparse
import importlib
import sys

import pandas as pd
import yaml
from tqdm import tqdm

spec = importlib.util.spec_from_file_location("tokenizer", "embeddings/tokenizer.py")
tokenizer = importlib.util.module_from_spec(spec)
sys.modules["tokenizer"] = tokenizer
spec.loader.exec_module(tokenizer)


def read_config(config_file):
    """
    Read YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary with configuration parameters.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def preprocess(df):
    """
    Preprocess the dates in MIMIC dataset.

    Args:
        df (pd.DataFrame): MIMIC dataset.

    Returns:
        pd.DataFrame: Preprocessed MIMIC dataset.
    """
    df = df.dropna(subset=["patient_text"])
    df = df.rename(columns={"patient_text": "text"})

    processed_rows = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        processed_row = tokenizer.preprocess_data(row, include_death_date=True)
        processed_rows.append(processed_row)

    df = pd.DataFrame(processed_rows)
    return df


def main(config_file):
    """
    Main function to create labels for the MIMIC dataset.

    Args:
        config_file (str): Path to the config file.
    """
    # Read configuration
    config = read_config(config_file)

    df = pd.read_parquet(config["paths"]["training_data"])
    df = preprocess(df)

    df["gender"] = df["text"].apply(lambda x: "M" if x.split()[6] == "8507" else "F")
    df["text_len"] = df["text"].apply(lambda x: len(x.split()))
    df = df[df["text_len"] > 9]
    df["visit_count"] = df["text"].apply(lambda x: x.count("[SEP]") - 1)
    df["n_tokens"] = df["text_len"] + 1
    df["n_tokens"] = df["n_tokens"].apply(lambda x: 512 if x > 512 else x)
    df["visit_count_trunc"] = df.apply(
        lambda x: (
            x["visit_count"]
            if x["n_tokens"] == x["text_len"] + 1
            else x["text"][: x["n_tokens"]].count("[SEP]")
        ),
        axis=1,
    )

    df.to_parquet(config["paths"]["mimic_df_labels"], compression="gzip")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Create labels for the MIMIC dataset.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    args = parser.parse_args()

    # Run the main function with the config file provided
    main(args.config)
