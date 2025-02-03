# GRU_module.py
import torch
import torch.nn as nn


class GRU_module(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.1,
    ):
        super(GRU_module, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim)
        batch_size = x.shape[0]
        num_nodes = x.shape[2]

        x = x.transpose(1, 2)
        # x: (batch_size, num_nodes, in_steps, input_dim)

        x = x.reshape(batch_size * num_nodes, -1, self.input_dim)
        # x: (batch_size * num_nodes, in_steps, input_dim)
        out, _ = self.gru(x)
        # out: (batch_size * num_nodes, in_steps, hidden_dim)

        out = out.reshape(batch_size, num_nodes, -1, self.hidden_dim)
        # out: (batch_size, num_nodes, in_steps, hidden_dim)
        out = out.transpose(1, 2)
        # out: (batch_size, in_steps, num_nodes, hidden_dim)

        return out