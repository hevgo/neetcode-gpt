import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        raw_q = self.q_proj(x) # B, T, num_heads * head_dim
        raw_k = self.k_proj(x) # B, T, num_kv_heads * head_dim
        raw_v = self.v_proj(x) # B, T, num_kv_heads * head_dim

        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        q = raw_q.view(B, T, self.num_heads, self.head_dim) # B, T, num_heads, head_dim
        k = raw_k.view(B, T, self.num_kv_heads, self.head_dim) # B, T, num_kv_heads, head_dim
        v = raw_v.view(B, T, self.num_kv_heads, self.head_dim) # B, T, num_kv_heads, head_dim

        q = q.transpose(1, 2) # B, num_heads, T head_dim
        k = k.transpose(1, 2) # B, num_kv_heads, T head_dim
        v = v.transpose(1, 2) # B, num_kv_heads, T head_dim

        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        num_repeats = num_heads // num_kv_heads
        k = torch.repeat_interleave(k, repeats=num_repeats, dim=1)
        v = torch.repeat_interleave(v, repeats=num_repeats, dim=1)

        # 4. Compute scaled dot-product attention with causal mask
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T,T, device = x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim = -1)
        output = weights @ v # B, num_heads, T, head_dim
        
        # 5. Concatenate heads and apply output projection
        output = output.transpose(1,2).contiguous() # B, T, num_heads, head_dim
        output = output.view(B, T, -1)
        output = self.output_proj(output)
        # 6. Return rounded output (decimals=4)
        return (torch.round(output, decimals=4))
