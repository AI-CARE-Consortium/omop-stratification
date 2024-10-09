import argparse
import importlib
import os
import sys

import pandas as pd
import yaml
from datasets import Features, Value
from tqdm import tqdm

spec = importlib.util.spec_from_file_location("tokenizer", 'embeddings/tokenizer.py')
tokenizer = importlib.util.module_from_spec(spec)
sys.modules["tokenizer"] = tokenizer
spec.loader.exec_module(tokenizer)


# Function to read YAML config file
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


# Function to read data with specified separator
def read_data(path, sep):
    """
    Read data from CSV file.

    Args:
        path (str): Path to the CSV file.
        sep (str): Separator.

    Returns:
        pd.DataFrame: DataFrame with data.
    """
    return pd.read_csv(path, sep=sep)


# Function to ensure datetime format
def ensure_datetime_format(df, date_columns):
    """
    Ensure datetime format in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with data.
        date_columns (list): List of date columns.

    Returns:
        pd.DataFrame: DataFrame with datetime format.
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# Function to aggregate patient data
def aggregate_patient_data(
    person,
    death,
    visit_occurrence,
    condition_occurrence,
    procedure_occurrence,
    drug_exposure,
    measurement,
    obseration,
):
    """
    Aggregate patient data.

    Args:
        person (pd.DataFrame): DataFrame with person data.
        death (pd.DataFrame): DataFrame with death data.
        visit_occurrence (pd.DataFrame): DataFrame with visit occurrence data.
        condition_occurrence (pd.DataFrame): DataFrame with condition occurrence data.
        procedure_occurrence (pd.DataFrame): DataFrame with procedure occurrence data.
        drug_exposure (pd.DataFrame): DataFrame with drug exposure data.
        measurement (pd.DataFrame): DataFrame with measurement data.
        obseration (pd.DataFrame): DataFrame with observation data.

    Returns:
        list: List of aggregated patient data.
    """
    patient_data = []
    death = ensure_datetime_format(death, ["death_date"])
    visit_occurrence = ensure_datetime_format(
        visit_occurrence, ["visit_start_date", "visit_end_date"]
    )

    for _, patient in tqdm(
        person.iterrows(), total=person.shape[0], desc="Aggregating patient data"
    ):
        person_id = patient["person_id"]
        year_of_birth = str(patient["year_of_birth"])
        gender_concept_id = patient["gender_concept_id"]
        ethnicity_concept_id = patient["ethnicity_concept_id"]
        race_concept_id = patient["race_concept_id"]

        death_row = death[death["person_id"] == person_id]
        death_date = (
            death_row["death_date"].dt.date.iloc[0] if not death_row.empty else None
        )
        death_cause_concept_id = (
            death_row["cause_concept_id"].iloc[0] if not death_row.empty else None
        )

        death_date = death_date.strftime("%Y-%m-%d") if death_date else None

        patient_visits = []
        all_concepts = []

        for _, visit in visit_occurrence[
            visit_occurrence["person_id"] == person_id
        ].iterrows():
            visit_occurrence_id = visit["visit_occurrence_id"]

            visit_start_date = (
                visit["visit_start_date"].strftime("%Y-%m-%d")
                if not pd.isnull(visit["visit_start_date"])
                else None
            )
            visit_end_date = (
                visit["visit_end_date"].strftime("%Y-%m-%d")
                if not pd.isnull(visit["visit_end_date"])
                else None
            )

            concepts = (
                condition_occurrence[
                    condition_occurrence["visit_occurrence_id"] == visit_occurrence_id
                ]["condition_concept_id"].tolist()
                + procedure_occurrence[
                    procedure_occurrence["visit_occurrence_id"] == visit_occurrence_id
                ]["procedure_concept_id"].tolist()
                + drug_exposure[
                    drug_exposure["visit_occurrence_id"] == visit_occurrence_id
                ]["drug_concept_id"].tolist()
                + measurement[
                    measurement["visit_occurrence_id"] == visit_occurrence_id
                ]["measurement_concept_id"].tolist()
                + obseration[obseration["visit_occurrence_id"] == visit_occurrence_id][
                    "observation_concept_id"
                ].tolist()
            )
            concepts = set(concepts)

            visit_data = {
                "visit_occurrence_id": visit_occurrence_id,
                "visit_start_date": visit_start_date,
                "visit_end_date": visit_end_date,
                "concepts": list(concepts),
            }

            patient_visits.append(visit_data)
            all_concepts += list(concepts)

        patient_object = {
            "person_id": person_id,
            "year_of_birth": year_of_birth,
            "gender_concept_id": gender_concept_id,
            "ethnicity_concept_id": ethnicity_concept_id,
            "race_concept_id": race_concept_id,
            "death_date": death_date,
            "all_concepts": list(set(all_concepts)),
            "death_cause_concept_id": death_cause_concept_id,
            "visits": patient_visits,
        }

        patient_data.append(patient_object)

    return patient_data


def construct_bert_inputs(aggregated_data):
    """
    Construct BERT inputs from aggregated data.

    Args:
        aggregated_data (list): List of aggregated patient data.

    Returns:
        list: List of BERT inputs.
    """
    bert_inputs = []

    for patient in aggregated_data:
        patient_text_parts = [
            f"[DATE] {patient['year_of_birth']}",
            f"[DATE] {patient['death_date'] if patient['death_date'] else 'None'}",
            f"[CODES] {patient['death_cause_concept_id'] if patient['death_cause_concept_id'] else 'None'}",
            str(patient["gender_concept_id"]),
            str(patient["ethnicity_concept_id"]),
            "[SEP]",
        ]

        visit_infos = []
        for visit in patient["visits"]:
            visit_info = f"[DATE] {visit['visit_start_date']} [DATE] {visit['visit_end_date']} [CODES] {' '.join(map(str, visit['concepts']))} [SEP]"
            visit_infos.append(visit_info)

        patient_text_parts.append(" ".join(visit_infos))
        patient_text = " ".join(patient_text_parts)

        bert_inputs.append(
            {"person_id": patient["person_id"], "patient_text": patient_text}
        )

    return bert_inputs


def dates_preprocess(df):
    """
    Preprocess dates in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with data.

    Returns:
        pd.DataFrame: DataFrame with preprocessed dates.
    """
    processed_rows = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        processed_row = tokenizer.preprocess_data(row, include_death_date=True)
        processed_rows.append(processed_row)

    return pd.DataFrame(processed_rows)


def create_labels(df):
    """
    Create labels for the data.

    Args:
        df (pd.DataFrame): DataFrame with preprocessed data.

    Returns:
        pd.DataFrame: DataFrame with labels.
    """
    df["gender"] = df["text"].apply(lambda x: "M" if x.split()[6] == "8507" else "F")
    df["text_len"] = df["text"].apply(lambda x: len(x.split()))
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
    return df

def tokenize(df, vocab_path):
    """
    Tokenize the text data.

    Args:
        df (pd.DataFrame): DataFrame with data.
        vocab_path (str): Path to the vocabulary file.

    Returns:
        datasets.Dataset: Dataset with tokenized data.
    """
    fixed_vocab_tokenizer = tokenizer.FixedVocabTokenizer(
        tokenizer.load_vocab(vocab_path),
        max_len=512,
        return_special_tokens_mask=True,
    )
    fixed_vocab_tokenizer.add_special_tokens(
        {
            "unk_token": "None",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "additional_special_tokens": ["[CODES]", "[DATE]"],
        }
    )
    test_df = df[["text"]]
    test_df.reset_index(drop=True, inplace=True)
    test_ds = df.from_pandas(
        test_df, split="test", features=Features({"text": Value("string")})
    )
    print("Datasets created. Example: ", test_ds[0])
    print("Vocab size: ", fixed_vocab_tokenizer.vocab_size)

    def tokenize_function(examples):
        examples["text"] = ["[CLS] " + text for text in examples["text"]]
        return {
            k: v.to("cuda")
            for k, v in tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=True,
            ).items()
        }
    test_ds = test_ds.map(tokenize_function, batched=True)
    return test_ds

def main(config_file):
    """
    Main function to load and process data.

    Args:
        config_file (str): Path to the config file.
    """
    # Read configuration
    config = read_config(config_file)

    # Base data path
    data_path = config["paths"]["cancer_omop"]

    # Read CSV files
    observation = read_data(
        os.path.join(data_path, "observation.csv"),
        sep=config["delimiters"]["observation"],
    )
    # observation_period = read_data(
    #     os.path.join(data_path, "observation_period.csv"),
    #     sep=config["delimiters"]["observation_period"],
    # )
    person = read_data(
        os.path.join(data_path, "person.csv"), sep=config["delimiters"]["person"]
    )
    condition_occurrence = read_data(
        os.path.join(data_path, "condition_occurrence.csv"),
        sep=config["delimiters"]["condition_occurrence"],
    )
    drug_exposure = read_data(
        os.path.join(data_path, "drug_exposure.csv"),
        sep=config["delimiters"]["drug_exposure"],
    )
    measurement = read_data(
        os.path.join(data_path, "measurement.csv"),
        sep=config["delimiters"]["measurement"],
    )
    procedure_occurrence = read_data(
        os.path.join(data_path, "procedure_occurrence.csv"),
        sep=config["delimiters"]["procedure_occurrence"],
    )
    visit_occurrence = read_data(
        os.path.join(data_path, "visit_occurrence.csv"),
        sep=config["delimiters"]["visit_occurrence"],
    )
    death = read_data(
        os.path.join(data_path, "death.csv"), sep=config["delimiters"]["death"]
    )

    # Preprocess dates (clean up)
    if "year_of_birth" in person.columns:
        person["year_of_birth"] = person["year_of_birth"].apply(
            lambda x: int(str(x)[-2:]) + 1900 if len(str(x)) == 5 else x
        )

    # Rename columns (wrong column names in the data)
    measurement = measurement.rename(
        columns={"visit_occurence_id": "visit_occurrence_id"}
    )
    procedure_occurrence = procedure_occurrence.rename(
        columns={"visit_occurence_id": "visit_occurrence_id"}
    )
    drug_exposure = drug_exposure.rename(
        columns={"visit_occurence_id": "visit_occurrence_id"}
    )
    observation = observation.rename(
        columns={"visit_occurence_id": "visit_occurrence_id"}
    )

    # Aggregating patient data
    aggregated_data = aggregate_patient_data(
        person,
        death,
        visit_occurrence,
        condition_occurrence,
        procedure_occurrence,
        drug_exposure,
        measurement,
        observation,
    )

    bert_inputs = pd.DataFrame(construct_bert_inputs(aggregated_data))
    bert_inputs = bert_inputs.dropna(subset=["patient_text"])
    bert_inputs = bert_inputs.rename(columns={"patient_text": "text"})

    bert_inputs_preprocessed_dates = dates_preprocess(bert_inputs)

    bert_inputs_preprocessed_dates_with_labels = create_labels(
        bert_inputs_preprocessed_dates
    )

    bert_inputs_preprocessed_dates_with_labels.to_parquet(
        config["paths"]["cancer_df_labels"], compression="gzip"
    )

    cancer_test_dataset = tokenize(bert_inputs_preprocessed_dates_with_labels, config["paths"]["tokenizer"] + "/vocab.json")

    cancer_test_dataset.save_to_disk(
        config["paths"]["cancer_dataset"]
    )


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Load and process data from CSV files."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    args = parser.parse_args()

    # Run the main function with the config file provided
    main(args.config)
