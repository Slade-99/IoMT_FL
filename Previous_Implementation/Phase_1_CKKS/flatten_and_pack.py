import torch
import math
from typing import List
import numpy as np


def model_to_numpy_vector(model: torch.nn.Module) -> np.ndarray:
    params = []
    for p in model.parameters():
        params.append(p.detach().cpu().view(-1).numpy())
    vec = np.concatenate(params, axis=0)
    return vec

def chunk_vector(vec: np.ndarray, slot_count: int) -> List[np.ndarray]:
    """Split vec into chunks each fitting slot_count"""
    L = len(vec)
    chunks = []
    i = 0
    while i < L:
        end = min(i + slot_count, L)
        chunk = vec[i:end]
        # pad to full slot length with zeros to keep operations consistent
        if len(chunk) < slot_count:
            pad = np.zeros(slot_count - len(chunk), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        chunks.append(chunk)
        i = end
    return chunks
