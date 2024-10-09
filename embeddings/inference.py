import gc
import subprocess as sp
from tqdm.auto import tqdm
import torch
import datasets
import yaml
import argparse
from transformers import BertForPreTraining


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


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


def main(config_file):
    # Read configuration
    config = read_config(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print("Using CPU! Finishing.")
        quit()
    print("Using device:", device)

    torch.cuda.empty_cache()

    print("Memory available:", get_gpu_memory())

    tokenized_dataset = datasets.load_from_disk(config["inference"]["tokeized_data_path"])
    model = BertForPreTraining.from_pretrained(
        config["inference"]["model_checkpoint"],
        output_hidden_states=True,
    )

    torch.cuda.empty_cache()
    total_memory = get_gpu_memory()[0]
    print("Memory on device:", total_memory, "MB")

    model.to(device)
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "token_type_ids"], device="cuda"
    )
    model.eval()
    print("Memory left:", total_memory - get_gpu_memory()[0], "MB")

    batch_size = 64
    total_steps = len(tokenized_dataset) // batch_size + (
        0 if len(tokenized_dataset) % batch_size == 0 else 1
    )
    progress = tqdm(
        range(0, len(tokenized_dataset), batch_size), total=total_steps, desc="Processing"
    )

    cls_list, mean_list = [], []

    for i in progress:
        progress.set_description(f"Processing ({get_gpu_memory()[0]}/{total_memory} MB)")

        batch = tokenized_dataset[i : i + batch_size]

        progress.set_description(f"Processing ({get_gpu_memory()[0]}/{total_memory} MB)")

        with torch.no_grad():
            outputs = model(**batch)

            progress.set_description(
                f"Processing ({get_gpu_memory()[0]}/{total_memory} MB)"
            )

            last_hidden_state = outputs.hidden_states[-1]

            # Mean embeddings
            attention_mask = (
                batch["attention_mask"].unsqueeze(-1).expand_as(last_hidden_state)
            )
            masked_hidden_state = last_hidden_state * attention_mask
            sum_embeddings = torch.sum(masked_hidden_state, dim=1)
            non_padding_tokens = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_batch = sum_embeddings / non_padding_tokens
            mean_batch = mean_batch.to("cpu")

            # CLS embeddings
            cls_batch = last_hidden_state[:, 0, :].to("cpu")

            cls_list.append(cls_batch)
            mean_list.append(mean_batch)

        del batch, outputs, last_hidden_state, cls_batch, mean_batch
        gc.collect()
        torch.cuda.empty_cache()

    cls = torch.cat(cls_list, 0)
    mean = torch.cat(mean_list, 0)

    print("Shape of CLS and mean: ", cls.shape, mean.shape)

    torch.save(cls, config["inference"]["cls_save_path"])
    torch.save(mean, config["inference"]["mean_save_path"])

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Running inference on the datasets to extract embeddings."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    args = parser.parse_args()

    # Run the main function with the config file provided
    main(args.config)
