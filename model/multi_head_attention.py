import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)

        # self.proj_k = nn.Linear(embedding_dim, attention_dim, bias = False)
        # self.proj_q = nn.Linear(embedding_dim, attention_dim, bias = False)
        # self.proj_v = nn.Linear(embedding_dim, attention_dim, bias = False)

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads # verification

        self.multi_heads = nn.ModuleList([self.SingleHeadAttention(self.embedding_dim, self.head_dim) for _ in range(num_heads)])
        self.projection = nn.Linear(attention_dim, attention_dim, bias = False)
        
        return

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places
        B, T, C = embedded.shape
        
        multi_head = [head(embedded) for head in self.multi_heads] # num_heads * B, T, head_dim
        concat_head = torch.cat(multi_head, dim = -1) # B, T, attention_dim
        projection = self.projection(concat_head)


        return torch.round(projection, decimals=4)



        

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v

    # def forward(self, embedded: TensorType[float]) -> TensorType[float]:
    #     # Run each head on the input, concatenate outputs along dim=2
    #     # Pass concatenated result through the output projection (W_O)
    #     # Return result rounded to 4 decimal places
    #     B, T, C = embedded.shape
        
    #     raw_k = self.proj_k(embedded) # B, T, attention_dim
    #     raw_q = self.proj_q(embedded)
    #     raw_v = self.proj_v(embedded)

    #     # split k, q, v
    #     k = raw_k.view(B, T, self.num_heads, self.head_dim) # B, T, num_head, head_dim
    #     q = raw_q.view(B, T, self.num_heads, self.head_dim)
    #     v = raw_v.view(B, T, self.num_heads, self.head_dim)

    #     # transpose
    #     k = k.transpose(1,2) # B, num_heads, T, head_dim
    #     q = q.transpose(1,2)
    #     v = v.transpose(1,2)

    #     # into multi head
    #     scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
    #     mask = torch.tril(torch.ones(T, T, device = embedded.device))
    #     scores = scores.masked_fill(mask == 0, float('-inf'))
    #     weights = torch.softmax(scores, dim=-1)
    #     output = weights @ v # B, num_head, T, head_dim
    #     output = output.transpose(1,2) # B, T, num_head, head_dim
    #     output = output.reshape(B, T, -1) # # B, T, attention_dim
    #     output = output.contiguous()
    #     projection = self.projection(output)

    #     return torch.round(projection, decimals=4)
