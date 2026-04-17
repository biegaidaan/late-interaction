import argparse
from omegaconf import OmegaConf
import os

import torch

from encode import Encoder
from common import registry

# Import to trigger @registry.register_* decorators
from models import *
from tokenizers import *


def parse_args():
    parser = argparse.ArgumentParser(description="Encoding")
    parser.add_argument("--config_path", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args


def load_texts(path: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.split("\t")[1].strip()  # format: "id\ttext"
            texts.append(text)
    return texts


def save(repr: dict, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    for k, v in repr.items():
        torch.save(v, os.path.join(output_path, f"{k}.pt"))


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    model = registry.get_model_cls(config.model.name).from_pretrained(args.config_path)
    tokenizer = registry.get_tokenizer_cls(config.tokenizer.name).from_config(config.tokenizer)

    encoder = Encoder(model, tokenizer, device=config.run.device)

    texts = load_texts(config.run.dataset_path)

    file_name = os.path.basename(config.run.dataset_path)
    if file_name == "collections.tsv":
        encoding = encoder.encode_doc(texts, int(config.run.bsize))
    elif file_name == "queries.tsv":
        encoding = encoder.encode_qry(texts, int(config.run.bsize))
    else:
        raise ValueError("Input file must be either collections.tsv or queries.tsv")

    save(encoding, config.run.output_path)
