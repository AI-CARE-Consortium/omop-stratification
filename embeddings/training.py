import random
import re
import subprocess as sp
import torch.multiprocessing as mp
import argparse
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertForPreTraining,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import datasets
from tokenizer import load_vocab, FixedVocabTokenizer
import warnings

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


def get_gpu_memory():
    """
    Get the current available GPU memory

    Returns:
    A list of the available GPU memory
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    result = sp.run(command.split(), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to get GPU memory info")
    lines = result.stdout.strip().split("\n")[1:]
    return [int(line.split()[0]) for line in lines]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_date(item):
    """
    Extracts the date from the section

    Args:
        item: The section to extract the date from

    Returns:
        A tuple containing the year, month, and day
    """
    match = re.search(r"\[DATE\] y(\d{2,3}) m(\d{2}) d(\d{2})", item)
    return tuple(map(int, match.groups())) if match else (0, 0, 0)


def main(config_path, load_tokenized_datasets):
    """
    Main function to train the model
    
    Args:
        config: Path to the config file
        load_tokenized_datasets: Load tokenized datasets
    """
    # Read configuration
    config = read_config(config_path)

    if not load_tokenized_datasets:
        data = pd.read_parquet(config["paths"]["training_data"])
        data["text"] = data["text"].str.replace("[SEP]", "[V]", regex=False)
        data["text"] = data["text"].apply(
            lambda text: text[:-4] if text.endswith(" [V]") else text
        )

        all_sections = data["text"].apply(lambda x: x.split(" [V] "))

        all_sections = all_sections.apply(
            lambda x: [x[0]] + sorted(x[1:], key=extract_date)
        )
        all_sections = all_sections.tolist()

        def create_nsp_pairs_for_row(sections):
            """Creates positive NSP pairs for a single row"""
            nsp_pairs = []
            labels = []

            # Create positive pairs
            for i in range(1, len(sections) - 1):
                sentence_a = " [V] ".join(sections[: i + 1]).strip()
                sentence_b = sections[i + 1].strip()
                nsp_pairs.append((sentence_a, sentence_b))
                labels.append(1)

            return nsp_pairs, labels

        def create_negative_nsp_pairs(all_sections):
            """Creates negative NSP pairs"""
            negative_pairs = []
            labels = []

            for sections in tqdm(
                all_sections,
                desc="Creating negative NSP pairs",
                ncols=80,
                ascii=True,
                position=0,
                leave=True,
            ):
                for i in range(1, len(sections) - 1):
                    date_a = extract_date(sections[i])
                    sentence_a = " [V] ".join(sections[: i + 1]).strip()
                    random_section = random.choice(all_sections)
                    sentence_b = random.choice(random_section[1:]).strip()
                    while random_section == sections:
                        print(
                            "Random section is the same as the current section, trying again..."
                        )
                        random_section = random.choice(all_sections)
                        sentence_b = random.choice(random_section[1:]).strip()
                        date_b = extract_date(sentence_b)
                        if (
                            date_a != (0, 0, 0)
                            and date_b != (0, 0, 0)
                            and date_b < date_a
                        ):
                            print(
                                "Date of sentence_b is earlier than date of sentence_a, trying again..."
                            )
                            continue

                    negative_pairs.append((sentence_a, sentence_b))
                    labels.append(0)

            return negative_pairs, labels

        nsp_data = []
        nsp_labels = []
        for sections in tqdm(all_sections, desc="Creating positive NSP pairs"):
            pairs, labels = create_nsp_pairs_for_row(sections)
            nsp_data.extend(pairs)
            nsp_labels.extend(labels)

        neg_pairs, neg_labels = create_negative_nsp_pairs(all_sections)
        nsp_data.extend(neg_pairs)
        nsp_labels.extend(neg_labels)

        df_nsp = pd.DataFrame(nsp_data, columns=["text_a", "text_b"])
        df_nsp["label"] = nsp_labels

        def check_negative_pairs(negative_pairs):
            """Checks the negative pairs for the condition sentence_b < sentence_a"""
            count = 0
            total = len(negative_pairs)
            for sentence_a, sentence_b in negative_pairs:
                date_a = extract_date(sentence_a.split(" [V] ")[-1])
                date_b = extract_date(sentence_b)
                if date_b < date_a:
                    count += 1
            return count, total

        neg_count, neg_total = check_negative_pairs(neg_pairs)
        neg_percentage = (neg_count / neg_total) * 100

        print(
            f"Number of negative pairs with sentence_b earlier than sentence_a: {neg_count}"
        )
        print(f"Total number of negative pairs: {neg_total}")
        print(f"Percentage of such negative pairs: {neg_percentage:.2f}%")

    vocab = load_vocab(config["paths"]["tokenizer"] + "/vocab.json")
    vocab["[V]"] = len(vocab)

    tokenizer = FixedVocabTokenizer(
        vocab=vocab,
        max_len=512,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    tokenizer.add_special_tokens(
        special_tokens_dict={
            "unk_token": "None",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "additional_special_tokens": ["[CODES]", "[DATE]", "[V]"],
        },
        replace_additional_special_tokens=False,
    )

    print(tokenizer.vocab_size)
    print(tokenizer.special_tokens_map)
    print(tokenizer.additional_special_tokens)

    if not load_tokenized_datasets:
        train_df, eval_df = train_test_split(
            df_nsp, train_size=config["bert"]["train_size"], random_state=42
        )
        training_dataset = datasets.Dataset.from_pandas(train_df)
        evaluation_dataset = datasets.Dataset.from_pandas(eval_df)
        def tokenize_function(example):
            """Tokenizes the text_a and text_b columns of the example"""
            encoding = tokenizer(
                example["text_a"],
                example["text_b"],
                padding="max_length",
                return_tensors="pt",
                max_length=512,
                verbose=False,
                truncation="longest_first",
                return_overflowing_tokens=False,
            )
            encoding["next_sentence_label"] = example["label"]
            return encoding

        tokenized_training_dataset = training_dataset.map(
            tokenize_function, batched=True
        )
        tokenized_evaluation_dataset = evaluation_dataset.map(
            tokenize_function, batched=True
        )
        tokenized_training_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "next_sentence_label",
            ],
        )
        tokenized_evaluation_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "next_sentence_label",
            ],
        )
        tokenized_training_dataset.save_to_disk(
           config["paths"]["tokenized_training_dataset"]
        )
        tokenized_evaluation_dataset.save_to_disk(
           config["paths"]["tokenized_evaluation_dataset"]
        )
    else:
        tokenized_evaluation_dataset = datasets.load_from_disk(
           config["paths"]["tokenized_training_dataset"]
        )
        tokenized_training_dataset = datasets.load_from_disk(
           config["paths"]["tokenized_evaluation_dataset"]
        )

    tokenized_training_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "next_sentence_label",
        ],
        device="cuda",
    )
    tokenized_evaluation_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "next_sentence_label",
        ],
        device="cuda",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config["bert"]["mlm_probability"]
    )

    model_config = BertConfig(
        vocab_size=tokenizer.vocab_size,
    )

    model = BertForPreTraining(model_config)

    def train_with_dynamic_batch_size():
        train_batch_size = config["bert"]["train_batch_size"]
        eval_batch_size = config["bert"]["eval_batch_size"]
        while train_batch_size > 0:
            try:
                training_args = TrainingArguments(
                    output_dir=config["bert"]["path_to_output_dir"],
                    max_steps=config["bert"]["max_training_steps"],
                    per_device_train_batch_size=train_batch_size,
                    per_device_eval_batch_size=eval_batch_size,
                    dataloader_num_workers=4,
                    dataloader_pin_memory=True,
                    overwrite_output_dir=True,
                    fp16=True,
                    save_total_limit=2,
                    prediction_loss_only=True,
                    logging_dir=config["bert"]["path_to_logs"],
                    logging_steps=config["bert"]["logging_steps"],
                    save_steps=config["bert"]["save_steps"],
                    evaluation_strategy="steps",
                    eval_steps=config["bert"]["eval_steps"],
                    eval_accumulation_steps=config["bert"]["eval_accumulation_steps"],
                    load_best_model_at_end=True,
                    metric_for_best_model="loss",
                    greater_is_better=False,
                    include_num_input_tokens_seen=True,
                    include_tokens_per_second=True,
                    warmup_ratio=config["bert"]["warmup_ratio"]
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=tokenized_training_dataset,
                    eval_dataset=tokenized_evaluation_dataset,
                    data_collator=data_collator,
                )

                print(trainer.args.parallel_mode)

                res = trainer.train()

                print(f"Training with batch size {train_batch_size} succeeded.")
                print(res)

                trainer.save_model(training_args.output_dir)

                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"Out of memory error with batch size {train_batch_size}. Reducing batch size and retrying..."
                    )
                    train_batch_size = max(1, train_batch_size // 2)
                    eval_batch_size = max(1, eval_batch_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e

    train_with_dynamic_batch_size()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    mp.set_start_method("spawn", force=True)

    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Train the model on the MIMIC dataset."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "--load_tokenized_datasets", type=bool, default=False, help="Load tokenized datasets"
    )
    args = parser.parse_args()

    # Run the main function with the config file provided
    main(args.config, args.load_tokenized_datasets)
