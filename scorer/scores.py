import torch

from common.registry import registry
from scorer.utils import mv_score, sv_score


def _finalize(qry_repr: dict, doc_repr: dict, scores: torch.Tensor, pairwise: bool) -> torch.Tensor:
    """Sum scores, apply optional mask normalization and CLS fusion."""
    scores = scores.sum(-1)

    if "mv_mask" in qry_repr:
        scores = scores / qry_repr["mv_mask"].sum(-1, keepdim=True)

    if "cls_repr" in qry_repr and "cls_repr" in doc_repr:
        scores = scores + sv_score(qry_repr["cls_repr"], doc_repr["cls_repr"], pairwise)

    return scores


@registry.register_scorer("maxsim_sum")
def maxsim_sum(qry_repr: dict, doc_repr: dict, pairwise: bool = False, **kwargs) -> torch.Tensor:
    P = mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise)
    return _finalize(qry_repr, doc_repr, P.max(dim=-1).values, pairwise)


@registry.register_scorer("soft_maxsim_sum")
def soft_maxsim_sum(qry_repr: dict, doc_repr: dict, pairwise: bool = False, topk: int = 8, temperature: float = 0.05, **kwargs) -> torch.Tensor:
    P = mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise)
    k = min(topk, P.size(-1))
    scores = P.topk(k, dim=-1).values  # (..., n, k)
    scores = temperature * torch.logsumexp(scores / temperature, dim=-1)  # (..., n)
    return _finalize(qry_repr, doc_repr, scores, pairwise)


@registry.register_scorer("topk_maxsim_sum")
def topk_maxsim_sum(qry_repr: dict, doc_repr: dict, pairwise: bool = False, topk: int = 8, **kwargs) -> torch.Tensor:
    P = mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise)
    k = min(topk, P.size(-1))
    scores = P.topk(k, dim=-1).values.sum(-1)  # (..., n)
    return _finalize(qry_repr, doc_repr, scores, pairwise)
