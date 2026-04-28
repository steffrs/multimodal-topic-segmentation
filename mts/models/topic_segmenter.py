from __future__ import annotations
import os
import contextlib

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from mts.utils import load_json_data
from mts.models.audio_encoder import SiamesePretrainedAudioEncoder
from mts.models.tagger import Tagger


class TopicSegmenter(nn.Module):

    def __init__(self, semantic_model_checkpoint: str, acoustic_model_checkpoint: str, max_len: int,
                 intermediate_output_dim: int | None, freeze_audio_encoder: bool = False,
                 load_encoder_weights: bool = True):
        super().__init__()
        self._semantic_model_checkpoint = semantic_model_checkpoint
        if load_encoder_weights:
            self._semantic_encoder = AutoModel.from_pretrained(semantic_model_checkpoint)
        else:
            enc_config = AutoConfig.from_pretrained(semantic_model_checkpoint)
            self._semantic_encoder = AutoModel.from_config(enc_config)

        self._acoustic_model_checkpoint = acoustic_model_checkpoint
        self._freeze_audio_encoder = freeze_audio_encoder
        self._acoustic_encoder = SiamesePretrainedAudioEncoder(
            acoustic_model_checkpoint,
            projection_size=self._semantic_encoder.config.hidden_size,
            freeze_encoder=self._freeze_audio_encoder,
            load_encoder_weights=load_encoder_weights,
        )

        classifier_inp_dim = self.semantic_dim + self.acoustic_dim
        self._classifier = Tagger(input_dim=classifier_inp_dim, dropout=0.1, max_position_embeddings=max_len,
                                  intermediate_output_dim=intermediate_output_dim)
        self._max_sentences_at_once = 128
        self._max_audio_feat_at_once = 32

    @property
    def semantic_dim(self):
        return self._semantic_encoder.config.hidden_size

    @property
    def acoustic_dim(self):
        return self._acoustic_encoder.output_dim

    def forward(self, x: torch.Tensor,
                train_semantic_encoder: bool,  # Passed in from outside
                skip_first: bool):  # Passed in from outside to avoid GPU sync in forward
        # x: (semantic, acoustic)
        # semantic: {"input_ids": ..., "attention_mask": ...}
        # acoustic: Tensor of shape (batch, num_sentences, 2, wave_len)
        x_semantic, x_acoustic = x

        bs, num_sents, len_sents = x_semantic["input_ids"].size()

        # Extract semantic inputs
        input_ids = x_semantic["input_ids"]     # (bs, num_sents, len_sents)
        attention_mask = x_semantic["attention_mask"]

        # Process semantic features in chunks
        semantic_chunks = []
        semantic_context = contextlib.nullcontext() if train_semantic_encoder else torch.no_grad()
        with semantic_context:
            for start_idx in range(0, num_sents, self._max_sentences_at_once):
                end_idx = start_idx + self._max_sentences_at_once
                # Slice the semantic input
                chunk_input_ids = input_ids[:, start_idx:end_idx, :].reshape(-1, len_sents)
                chunk_attention_mask = attention_mask[:, start_idx:end_idx, :].reshape(-1, len_sents)

                # Forward pass through semantic encoder
                chunk_x_sem = self._semantic_encoder(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask
                )
                chunk_x_sem = torch.mean(chunk_x_sem.last_hidden_state, dim=1)
                semantic_chunks.append(chunk_x_sem)

        # Concatenate all semantic chunks: shape [num_sents, hidden_dim]
        x_semantic = torch.cat(semantic_chunks, dim=0).unsqueeze(0)

        # Process acoustic features
        # x_acoustic: (batch, num_sents, 2, wave_len), typically batch=1
        x_acoustic = x_acoustic.squeeze(0)  # shape: (num_sents, 2, wave_len)

        acoustic_chunks = []
        start = 1 if skip_first else 0
        acoustic_context = torch.no_grad() if train_semantic_encoder else contextlib.nullcontext()
        with acoustic_context:
            for start_idx in range(start, num_sents, self._max_audio_feat_at_once):
                end_idx = start_idx + self._max_audio_feat_at_once
                chunk_x_ac = x_acoustic[start_idx:end_idx, :, :]
                # Forward pass through acoustic encoder
                chunk_x_ac = self._acoustic_encoder(chunk_x_ac)
                acoustic_chunks.append(chunk_x_ac)

        # Concatenate acoustic chunks: shape [num_sents, acoustic_dim]
        x_acoustic = torch.cat(acoustic_chunks, dim=0)

        # If we skipped the first one, add a zero vector at the start
        if skip_first:
            audio_dim = x_acoustic.size(1)
            zeros_chunk = torch.zeros((1, audio_dim), dtype=x_acoustic.dtype, device=x_acoustic.device)
            x_acoustic = torch.cat([zeros_chunk, x_acoustic], dim=0)

        x_acoustic = x_acoustic.unsqueeze(0)

        x = torch.cat((x_semantic, x_acoustic), dim=2)
        x = self._classifier(x)
        return x

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> TopicSegmenter:
        config_path = os.path.join(checkpoint_dir, "config.json")
        train_config = load_json_data(config_path)
        semantic_checkpoint = train_config["semantic_model_checkpoint"]
        acoustic_checkpoint = train_config["acoustic_model_checkpoint"]
        max_len = train_config["max_len"]
        intermediate_output_dim = train_config.get("intermediate_output_dim", 768)
        freeze_audio_encoder = train_config.get("freeze_audio_encoder", False)

        model = cls(semantic_checkpoint, acoustic_checkpoint, max_len, intermediate_output_dim, freeze_audio_encoder,
                    load_encoder_weights=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(checkpoint_dir, "model.pt")
        model_weights = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(model_weights)
        return model
