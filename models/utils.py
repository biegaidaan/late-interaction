import torch


def sv_score(qry_repr: torch.Tensor, doc_repr: torch.Tensor, pairwise: bool = False) -> torch.Tensor:
    if pairwise:
        N = doc_repr.shape[0] // qry_repr.shape[0]
        doc_repr = doc_repr.view(-1, N, doc_repr.shape[-1])  # Q, N, d
        scores = (qry_repr.unsqueeze(1) * doc_repr).sum(-1)  # Q, N
    else:
        scores = qry_repr @ doc_repr.transpose(0, 1)  # Q, D
    return scores


def mv_score(qry_repr: torch.Tensor, doc_repr: torch.Tensor, pairwise: bool = False) -> torch.Tensor:
    if pairwise:
        N = doc_repr.shape[0] // qry_repr.shape[0]

        Q = qry_repr.unsqueeze(1)  # Q, 1, Lq, d
        D = doc_repr.view(-1, N, *doc_repr.shape[-2:])  # Q, N, Ld, d

        scores = torch.einsum("bild,bnjd->bnij", Q, D)  # Q, N, Lq, Ld
    else:  # inbatch negative sampling
        scores = torch.einsum("qik,djk->qdij", qry_repr, doc_repr)  # Q, D, Lq, Ld
    return scores


def maxsum(mv_scores: torch.Tensor) -> torch.Tensor:
    return mv_scores.max(-1).values.sum(-1)