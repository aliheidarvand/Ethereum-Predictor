import torch
import torch.nn as nn

class GRUTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        transformer_dim: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        if gru_hidden != transformer_dim:
            self.proj = nn.Linear(gru_hidden, transformer_dim)
        else:
            self.proj = nn.Identity()

        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, transformer_dim))
        nn.init.trunc_normal_(self.pos_encoder, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        self.fc = nn.Linear(transformer_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        t_in = self.proj(gru_out)
        seq_len = t_in.size(1)
        pe = self.pos_encoder[:, :seq_len, :]
        t_in = t_in + pe
        t_out = self.transformer(t_in)
        pooled = t_out.mean(dim=1)
        out = self.fc(pooled).squeeze(-1)
        return out
