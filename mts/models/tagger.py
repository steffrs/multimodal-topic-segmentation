import torch
import torch.nn as nn
from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerConfig


class Tagger(nn.Module):

    def __init__(self, input_dim: int, dropout: float, max_position_embeddings: int,
                 intermediate_output_dim: int | None = None):
        super().__init__()
        if input_dim % 8 != 0:
            self._pre_proj_layer = nn.Linear(in_features=input_dim, out_features=768)
            input_dim = 768
            self._use_pre_proj = True
        else:
            self._use_pre_proj = False
        self._roformer_config = RoFormerConfig(hidden_size=input_dim,
                                               num_hidden_layers=12,
                                               num_attention_heads=8,
                                               intermediate_size=2048,
                                               max_position_embeddings=max_position_embeddings,
                                               )
        self._roformer_encoder = RoFormerEncoder(self._roformer_config)
        self._intermediate_output_dim = intermediate_output_dim
        if intermediate_output_dim is not None:
            self._proj_layer = nn.Linear(in_features=input_dim, out_features=intermediate_output_dim)
            self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        output_in_dim = input_dim if intermediate_output_dim is None else intermediate_output_dim
        self._output_layer = nn.Linear(in_features=output_in_dim, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_pre_proj:
            x = self._pre_proj_layer(x)
        x = self._roformer_encoder(x)
        x = self._dropout_layer(x.last_hidden_state)
        if self._intermediate_output_dim is not None:
            x = self._proj_layer(x)
            x = self._relu(x)
            x = self._dropout_layer(x)
        x = self._output_layer(x)
        return x
