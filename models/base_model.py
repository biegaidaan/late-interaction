import os

import torch
import torch.nn as nn

from omegaconf import OmegaConf


class BaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, config_path: str):
        cfg = OmegaConf.load(config_path).model

        model = cls.from_config(cfg)

        if "ckpt_path" in cfg:
            if not os.path.exists(cfg.ckpt_path):
                raise FileNotFoundError(f"Checkpoint file '{cfg.ckpt_path}' does not exist")

            ckpt = torch.load(cfg.ckpt_path)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)

        return model


class BaseEncoder:
    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        raise NotImplementedError