import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer


def get_text(df):
    """
    Generator function to yield text from a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with text column.
        
    Yields:
        str: Text from the DataFrame.
    """
    for i in range(df.shape[0]):
        yield df.iloc[i].text


def build_vocab(texts, vocab):
    """
    Build a vocabulary from a list of texts.
    Args:
        texts (list): List of texts.
        vocab (dict): Vocabulary dictionary.

    Returns:
        dict: Updated vocabulary dictionary.
    """
    for text in texts:
        if text is not None:
            for token in text.split():
                if token.startswith("date_"):
                    date_token = token[5:]  # Remove 'date_' prefix
                    date_parts = tokenize_date(date_token)
                    for part in date_parts:
                        if part not in vocab:
                            vocab[part] = len(vocab)
                elif token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def save_vocab(vocab, path):
    """
    Save a vocabulary to a JSON file.

    Args:
        vocab (dict): Vocabulary dictionary.
        path (str): Path to save the vocabulary.
    """
    with open(path, "w") as f:
        for word, index in vocab.items():
            f.write(f"{word} {index}\n")


def load_vocab(path):
    """
    Load a vocabulary from a JSON file.

    Args:
        path (str): Path to load the vocabulary.

    Returns:
        dict: Vocabulary dictionary.
    """
    with open(path, "r") as f:
        vocab = json.load(f)
    return vocab


def load_vocab_from_txt(path):
    """
    Load a vocabulary from a text file.

    Args:
        path (str): Path to load the vocabulary.

    Returns:
        dict: Vocabulary dictionary.
    """
    with open(path, "r") as f:
        vocab = {}
        for line in f:
            word, index = line.split()
            vocab[word] = int(index)
    return vocab


def tokenize_date(date_str):
    """
    Helper function to tokenize dates.

    Args:
        date_str (str): Date string.

    Returns:
        list: List of tokens.
    """

    if date_str == "empty":
        return ["None", "None", "None"]

    if "-" in date_str:
        year, month, day = date_str.split("-")
        return ["y" + year, "m" + month, "d" + day]

    return ["y" + date_str, "None", "None"]


def get_relative_date(date_token, birth_year):
    """
    Helper function to get the relative date.
    
    Args:
        date_token (str): Date token.
        birth_year (int): Birth year.
        
    Returns:
        str: Relative date.
    """
    if not date_token or date_token == "None":
        return "None None None"
    year, month, day = date_token.split("-")
    if month == "None" or day == "None":
        print(f"Invalid date format: {date_token}")
    year_diff = int(year) - birth_year if year.isdigit() else 0
    return f"y{year_diff} m{month} d{day}"


def preprocess_data(row, include_death_date=False):
    """
    Preprocess a row of data.

    Args:
        row (pd.Series): Row of data.
        include_death_date (bool): Include death date or not.

    Returns:
        pd.Series: Processed row of data.
    """
    tokens = row["text"].split()
    birth_date, death_date = None, None
    birth_year = None
    processed_tokens = []
    date_count = 0
    death_age = None
    with tqdm(total=len(tokens), desc="Processing tokens", leave=False) as pbar:
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "[DATE]" and i + 1 < len(tokens):
                date_count += 1
                date_token = tokens[i + 1]
                date_rel_token = None

                if date_count == 1:
                    birth_date = date_token
                    birth_year = int(birth_date.split("-")[0])
                    i += 2
                    pbar.update(i - pbar.n)
                    continue
                if date_count == 2:
                    death_date = date_token
                    death_rel_date = get_relative_date(date_token, birth_year)
                    death_year = death_rel_date.split(" ")[0]
                    if death_year != "None":
                        death_age = int(death_year[1:])
                    if include_death_date:
                        date_rel_token = death_rel_date
                    else:

                        i += 2
                        pbar.update(i - pbar.n)
                        continue
                else:
                    date_rel_token = get_relative_date(date_token, birth_year)

                processed_tokens.append("[DATE]")
                processed_tokens.append(date_rel_token)

                i += 2
            else:
                processed_tokens.append(token)
                i += 1
            pbar.update(i - pbar.n)

    row["text"] = " ".join(processed_tokens)
    row["birth_date"] = birth_date
    row["death_date"] = death_date
    row["death_age"] = death_age
    return row


class FixedVocabTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer with a fixed vocabulary.

    Args:
        vocab (dict): Vocabulary dictionary.
        max_len (int): Maximum length of the tokenized text.
    """
    def __init__(self, vocab: Dict[str, int], max_len: int = None, **kwargs):
        self._token_ids = vocab
        self._id_tokens: Dict[int, str] = {value: key for key, value in vocab.items()}
        self.bos_token = None
        self.eos_token = None
        super().__init__(max_len=max_len, verbose=True, **kwargs)

    def _tokenize(self, text: str, **kwargs):
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_ids[token] if token in self._token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_tokens[index] if index in self._id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self._token_ids.copy()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ""
        vocab_path = Path(save_directory, filename_prefix + "vocab.json")
        json.dump(self._token_ids, open(vocab_path, "w"))
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)