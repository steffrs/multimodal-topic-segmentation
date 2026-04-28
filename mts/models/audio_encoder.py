from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SiamesePretrainedAudioEncoder(nn.Module):

    def __init__(self, model_checkpoint: str, projection_size: int | None = None,
                 freeze_encoder: bool = False, load_encoder_weights: bool = True):
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._acoustic_model = AutoModel.from_pretrained(model_checkpoint)

        if load_encoder_weights:
            self._acoustic_model = AutoModel.from_pretrained(model_checkpoint)
        else:
            enc_config = AutoConfig.from_pretrained(model_checkpoint)
            self._acoustic_model = AutoModel.from_config(enc_config)


        # We only use projection layer if really needed (output must be projected to meet dim requirements)
        self._use_projection = (projection_size is not None and
                                projection_size // 2 != self._acoustic_model.config.hidden_size)
        if self._use_projection:
            self._projector = nn.Linear(self._acoustic_model.config.hidden_size, projection_size // 2)
            self.output_dim = projection_size
        else:
            self.output_dim = self._acoustic_model.config.hidden_size
        self.activation = nn.ReLU()
        if freeze_encoder:
            print("- Siamese Audio Encoder: Freezing feature encoder", flush=True)
            for param in self._acoustic_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # batch_size x 2 x audio_feat_len
        batch_size, num_siblings, feat_len = x.size()
        x = x.view(batch_size * num_siblings, feat_len)  # resize to add siblings dim to batch dim (siamese network)
        x = self._acoustic_model(x).last_hidden_state  # take hidden_states from wav2vec2 output
        x = x.mean(dim=1)
        # return self.activation(x)
        if self._use_projection:
            x = self._projector(x)
        x = x.view(batch_size, -1)  # concat left and right contexts for each sample (bs x output_dim)
        return torch.tanh(x)
