import argparse
from omegaconf import OmegaConf
from dataclasses import dataclass
import os

import torch
from tqdm import tqdm

from encode import StrideTensor
from common import registry

# Import to trigger @registry.register_* decorators
import scorer


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config_path", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args


@dataclass
class Encodings:
    mv_repr: StrideTensor
    cls_repr: torch.Tensor | None = None

    def lookup(self, start: int, num: int = 1) -> dict[str, torch.Tensor]:
        result = {"mv_repr": self.mv_repr.lookup(start, num)}
        if self.cls_repr is not None:
            result["cls_repr"] = self.cls_repr[start: start + num]
        return result


def load_encoding(dir_path: str, device: str) -> dict[str, torch.Tensor | StrideTensor]:
    mv_repr_path = os.path.join(dir_path, "mv_repr.pt")
    mv_lens_path = os.path.join(dir_path, "mv_lens.pt")
    cls_repr_path = os.path.join(dir_path, "cls_repr.pt")

    encoding = {}

    if os.path.exists(mv_repr_path) and os.path.exists(mv_lens_path):
        encoding["mv_repr"] = StrideTensor.from_packed_tensor(
            torch.load(mv_repr_path, weights_only=False),
            torch.load(mv_lens_path, weights_only=False),
            device
        )
    else:
        raise FileNotFoundError(f"Missing mv_repr.pt or mv_lens.pt in {dir_path}")

    if os.path.exists(cls_repr_path):
        encoding["cls_repr"] = torch.load(cls_repr_path, weights_only=False).to(device)

    return Encodings(**encoding)


def retrieve(
    score_func_name: str,
    qry_encoding_dir: str,
    doc_encoding_dir: str,
    qry_bsize: int,
    doc_bsize: int,
    device: str,
    show_progress: bool = True,
    topk: int = 10,
    output_path: str | None = None
) -> None:
    Q = load_encoding(qry_encoding_dir, device)
    D = load_encoding(doc_encoding_dir, device)

    score_func = registry.get_scorer(score_func_name)

    results = {}
    qid = 0
    for i in tqdm(range(0, len(Q.mv_repr), qry_bsize), disable=not show_progress):
        Q_batch = Q.lookup(i, qry_bsize)

        all_scores = []
        for j in tqdm(range(0, len(D.mv_repr), doc_bsize), disable=not show_progress):
            D_batch = D.lookup(j, doc_bsize)

            scores = score_func(Q_batch, D_batch, pairwise=False)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, -1)
        topk_vals, topk_indices = torch.topk(all_scores, k=topk, dim=-1)

        for k in range(topk_indices.shape[0]):
            results[qid] = {str(idx.item()): float(val) for idx, val in zip(topk_indices[k], topk_vals[k])}
            qid += 1

    if output_path is not None:
        with open(output_path, "w") as f:
            import json
            json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    retrieve(**config)