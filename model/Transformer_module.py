import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask_type="none"):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask_type = mask_type  # "none", "temporal", "spatial", "mixed", "decoupled"

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask_type == "temporal":
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        elif self.mask_type == "spatial":
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).fill_diagonal_(0)
            attn_score.masked_fill_(~mask, -torch.inf)
        
        elif self.mask_type == "mixed":
            mask_t = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()
            mask_s = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).fill_diagonal_(0)
            mask = mask_t | mask_s
            attn_score.masked_fill_(~mask, -torch.inf)
        
        elif self.mask_type == "decoupled":
            # this mask allow attend only on the same time-step
             mask = torch.eye(tgt_length, src_length, dtype=torch.bool, device=query.device)
             attn_score.masked_fill_(~mask, -torch.inf)
        

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask_type="none"
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask_type)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class Transformer_module(nn.Module):
    def __init__(
        self,
        model_dim,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        dim = -2,
        mask_type = "none"
    ):
        super().__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers
        self.dim = dim
        self.mask_type = mask_type

        self.attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout, mask_type)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, model_dim)
        for attn in self.attn_layers:
            x = attn(x, dim=self.dim)
        return x