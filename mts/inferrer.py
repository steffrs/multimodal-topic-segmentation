import os

import torch
from transformers import AutoTokenizer, AutoProcessor, BatchEncoding

from mts.utils import load_json_data
from mts.load import load_from_sentences
from mts.models.topic_segmenter import TopicSegmenter


class Inferrer:

    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        config_path = os.path.join(self.checkpoint, "config.json")
        self.train_config = load_json_data(config_path)
        # Define max sequence length (num sents) to be processed at once
        self.max_len = self.train_config["max_len"]

        self._model = self._load_model()
        self._model.to(self._device)
        self._model.eval()
        self._softmax = torch.nn.Softmax(dim=1)

        self._tokenizer = AutoTokenizer.from_pretrained(self.train_config["semantic_model_checkpoint"])
        self._processor = AutoProcessor.from_pretrained(self.train_config["acoustic_model_checkpoint"])
        sample_rate = 16000
        self._max_num_audio_samples = int(self.train_config["chunk_seconds"] * sample_rate)

    def _load_model(self):
        return TopicSegmenter.from_checkpoint(self.checkpoint)

    def __call__(self, sentences: list[dict], audio_filepath: str, return_logits: bool = False) -> list[int | float]:
        input_samples = self._load_samples_from_sentences(sentences, audio_filepath)
        predictions = self.run_on_samples(input_samples, return_logits)
        # if self.cfg.skip_start:
        #     # Skip first prediction as we are dealing with topic starts, but need change labels
        #     return predictions[1:]  #, targets[1:]
        return predictions  #, targets

    def _load_samples_from_sentences(self, sentences: list[dict], audio_filepath: str) -> list:
        samples = []
        for text_feat, audio_feat in load_from_sentences(
                sentences=sentences,
                audio_filepath=audio_filepath,
                topic_start_tags=None,
                max_len=self.max_len,
                tokenizer=self._tokenizer,
                processor=self._processor,
                chunk_seconds=self.train_config["chunk_seconds"],
                boundary_seconds=self.train_config["boundary_seconds"],
        ):
            # load acoustic features
            # acoustic_features = load_audio_features_seq(features[1], processor=self._processor,
            #                                             max_num_samples=self._max_num_audio_samples)
            sample = (text_feat, audio_feat)
            samples.append(sample)
        return samples

    def run_on_samples(self, samples: list, return_logits: bool = False) -> list[int | float]:
        # todo: adapt (or create new one) to handle binary logits (only one value per sentence tag)
        all_predictions = []
        # all_tags = []
        for n, feat in enumerate(samples):
            logits = self._predict_for_sample(feat, n)
            if logits.shape[-1] == 2:
                # Trained using cross-entropy loss (one value per class)
                if return_logits:
                    logits_normalized = self._softmax(logits)
                    predictions = logits_normalized[:, 1]  # Use normalized scores for positive class
                else:
                    predictions = logits.argmax(dim=1)
            elif logits.shape[-1] == 1:
                # Trained using binary cross-entropy loss
                logits = torch.sigmoid(logits.squeeze(1))
                if return_logits:
                    predictions = logits.tolist()
                else:
                    predictions = (logits > 0.5).int().tolist()
            all_predictions.extend(predictions)
            # all_tags.extend(tags.tolist())
        return all_predictions

    def _predict_for_sample(self, sample, n: int) -> torch.Tensor:
        skip_first = n == 0
        with torch.no_grad():
            sample = self._add_batch_dimension(sample)
            logits = self._model(sample, train_semantic_encoder=False, skip_first=skip_first)
            logits = logits.detach().cpu()
        return logits.squeeze(0)

    def _add_batch_dimension(self, sample):
        if isinstance(sample, torch.Tensor):
            return sample.unsqueeze(0).to(self._device)
        if isinstance(sample, list):
            return [self._add_batch_dimension(itm) for itm in sample]
        if isinstance(sample, tuple):
            return tuple(self._add_batch_dimension(itm) for itm in sample)
        if isinstance(sample, dict) or isinstance(sample, BatchEncoding):
            return {k: self._add_batch_dimension(v) for k, v in sample.items()}
        raise TypeError(f"Type of sample '{type(sample)}' not supported for batch conversion")
