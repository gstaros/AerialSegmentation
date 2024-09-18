import torch

def one_hot_decoding(one_hot_matrix: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.argmax(one_hot_matrix, dim=dim)