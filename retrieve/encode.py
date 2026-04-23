import torch

from encode import Encoder
from retrieve.utils import save_encoding


def encode_queries(
    encoder: Encoder,
    queries: list[str],
    bsize: int | None = None,
    show_progress: bool = True,
    output_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """Encode queries and return the raw encoding dict.

    If output_path is provided, saves the encoding to that directory.
    """
    encoding = encoder.encode_qry(queries, bsize, show_progress)
    if output_path is not None:
        save_encoding(encoding, output_path)
    return encoding


def encode_docs(
    encoder: Encoder,
    docs: list[str],
    bsize: int | None = None,
    show_progress: bool = True,
    output_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """Encode documents and return the raw encoding dict.

    If output_path is provided, saves the encoding to that directory.
    """
    encoding = encoder.encode_doc(docs, bsize, show_progress)
    if output_path is not None:
        save_encoding(encoding, output_path)
    return encoding
