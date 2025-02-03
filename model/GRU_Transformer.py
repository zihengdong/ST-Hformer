import torch
import torch.nn as nn
from model.GRU_module import GRU_module  
from model.Transformer_module import Transformer_module  


class GRU_Transformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        fourier_embedding_dim=32,  
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        gru_hidden_dim=128,
        gru_num_layers = 2,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        mask_type = "none",
        num_harmonics=4  
    ):
        super().__init__()

        # Add num_harmonics as instance variable
        self.num_harmonics = num_harmonics
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim + (num_harmonics * 2 if tod_embedding_dim > 0 else 0)  
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
            + (fourier_embedding_dim if tod_embedding_dim > 0 else 0)  
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.mask_type = mask_type

      
        self.traffic_proj = nn.Linear(1, input_embedding_dim)  
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
            # Add linear projection for Fourier features
            self.fourier_proj = nn.Linear(num_harmonics * 2, fourier_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        
        if tod_embedding_dim > 0 and dow_embedding_dim > 0:
            self.temporal_fusion = nn.Sequential(
                nn.Linear(tod_embedding_dim + dow_embedding_dim, tod_embedding_dim + dow_embedding_dim),
                nn.ReLU(),
                nn.Linear(tod_embedding_dim + dow_embedding_dim, tod_embedding_dim + dow_embedding_dim),
            )
            
        self.gru_module = GRU_module(
            input_dim = self.model_dim,
            hidden_dim = gru_hidden_dim,
            num_layers = gru_num_layers,
            dropout = dropout
        )

        self.transformer_t_module = Transformer_module(
            model_dim = gru_hidden_dim,
            feed_forward_dim = feed_forward_dim,
            num_heads = num_heads,
            num_layers = num_layers,
            dropout = dropout,
            dim = 1,
            mask_type = mask_type
        )

        self.transformer_s_module = Transformer_module(
            model_dim = gru_hidden_dim,
            feed_forward_dim = feed_forward_dim,
            num_heads = num_heads,
            num_layers = num_layers,
            dropout = dropout,
            dim = 2,
            mask_type = mask_type
        )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * gru_hidden_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(gru_hidden_dim, self.output_dim)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow+fourier)
        batch_size = x.shape[0]
        
        # Split input features and reshape for projection
        traffic = x[..., 0:1]  # (batch_size, in_steps, num_nodes, 1)
        B, T, N, _ = traffic.shape
        
        # Process traffic data
        traffic = traffic.reshape(B * T * N, -1)  # Flatten for linear layer
        traffic = self.traffic_proj(traffic)  # Project traffic data
        traffic = traffic.reshape(B, T, N, -1)  # Reshape back
        features = [traffic]

        temporal_features = []
        if self.tod_embedding_dim > 0:
            # Process time of day
            tod = x[..., 1:2]
            tod = (tod * self.steps_per_day).long().squeeze(-1)  
            tod_emb = self.tod_embedding(tod)  # Will automatically add embedding dimension
            temporal_features.append(tod_emb)
            
            # Process Fourier features
            fourier_start = 3 if self.dow_embedding_dim > 0 else 2
            fourier_features = x[..., fourier_start:fourier_start+self.num_harmonics*2]
            fourier_features = fourier_features.reshape(B * T * N, -1)
            fourier_emb = self.fourier_proj(fourier_features)
            fourier_emb = fourier_emb.reshape(B, T, N, -1)
            features.append(fourier_emb)
        
        if self.dow_embedding_dim > 0:
            # Process day of week
            dow = x[..., 2:3]
            dow = dow.long().squeeze(-1)  
            dow_emb = self.dow_embedding(dow)  # Will automatically add embedding dimension
            temporal_features.append(dow_emb)


        if len(temporal_features) > 1:
            temporal_concat = torch.cat(temporal_features, dim=-1)
            temporal_fused = self.temporal_fusion(temporal_concat)
            features.append(temporal_fused)
        else:
            features.extend(temporal_features)
            
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(B, T, N, -1)
            features.append(spatial_emb)
            
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(B, -1, -1, -1)
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)
        
        x = self.gru_module(x)
        # x: (batch_size, in_steps, num_nodes, gru_hidden_dim)

        x = self.transformer_t_module(x)
        # x: (batch_size, in_steps, num_nodes, gru_hidden_dim)

        x = self.transformer_s_module(x)
        # x: (batch_size, in_steps, num_nodes, gru_hidden_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, gru_hidden_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.gru_hidden_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, gru_hidden_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, gru_hidden_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out