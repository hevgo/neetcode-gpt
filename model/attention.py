import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.k = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.q = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.v = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.attention_dim = attention_dim

        return

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        
        B, T, C = embedded.shape
        
        k = self.k(embedded)
        q = self.q(embedded)
        v = self.v(embedded)

        # print (k.shape)
        # print (q.shape)
        # print (v.shape)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.attention_dim)
        mask = torch.tril(torch.ones((T,T), device = embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        output = weights @ v

        return torch.round(output, decimals=4)
