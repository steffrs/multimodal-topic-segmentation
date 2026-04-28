from pathlib import Path
from typing import Generator

import torch
from transformers import AutoProcessor, AutoTokenizer
import torchaudio


# Input sample: transcript sentences with timestamps and audio path
# Input sentences: [{"text": "...", "start": float, "end": float}, ...]


MAX_TOKENS_PER_SENTENCE=128
TARGET_SAMPLE_RATE = 16000


def load_text_features(sentences: list[str], topic_start_tags: list[int], max_len: int, tokenizer: AutoTokenizer
                       ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    if len(sentences) < 2:
        # there is a video (id: vrJY85dBJLc) with only one sentence and one label. This will be skipped below.
        print(f"--> Total num sents of video <2, skipping sample.", flush=True)
        return None

    sent_transcripts, tags = [], []
    for n, sentence in enumerate(sentences):
        sent_transcripts.append(sentence)
        tags.append(topic_start_tags[n])

        if (n+1) % max_len == 0 or n == len(sentences) - 1:
            features = tokenizer(sent_transcripts, padding="longest", max_length=MAX_TOKENS_PER_SENTENCE,
                                 truncation=True, return_tensors="pt")  # Check if truncation would help (e.g. oom)
            tags = torch.tensor(tags)
            yield features, tags
            sent_transcripts, tags = [], []


def load_acoustic_feature_times(sentences: list[dict], chunk_seconds: float, boundary_seconds: float
                                ) -> list[tuple[tuple, int]]:
    """Returns list of (audio_path, start1, start2, num_samples_1, num_samples_2)"""
    samples = []
    sample_rate = TARGET_SAMPLE_RATE
    num_samples_chunk = int(chunk_seconds * sample_rate)
    num_samples_boundary = int(boundary_seconds * sample_rate)

    for n, sentence in enumerate(sentences[:-1]):
        # tag = topic_start_tags[n+1]
        next_sentence = sentences[n + 1]
        end_sent_1 = int(sentence["end"] * sample_rate)
        start_sent_2 = int(next_sentence["start"] * sample_rate)
        num_samples_between = int((next_sentence["start"] - sentence["end"]) * sample_rate)
        num_samples_boundary = min(num_samples_boundary, num_samples_between // 2)

        start_feat_1 = end_sent_1 - (num_samples_chunk - num_samples_boundary)
        start_feat_1 = max(0, start_feat_1)
        num_samples_feat_1 = min(num_samples_chunk, end_sent_1 + num_samples_boundary)
        start_feat_2 = start_sent_2 - num_samples_boundary
        start_feat_2 = max(0, start_feat_2)
        # if num_samples_feat_2 exceeds audio file, it will be using all possible samples
        # start_feat_1, start_feat_2, num_chunk_samples_1, num_chunk_samples_2 = start_feat_1, start_feat_2, num_samples_feat_1, num_samples_chunk

        features = (start_feat_1, start_feat_2, num_samples_feat_1, num_samples_chunk)
        samples.append(features)
    return samples


def _create_batched_padded_features(chunks: list, max_num_samples: int, padding_side: str = "right") -> torch.Tensor:
    max_length = max(chunks, key=lambda x: x.shape[0]).shape[0]
    max_num_samples = min(max_length, max_num_samples)
    padded = torch.zeros(len(chunks), max_num_samples, dtype=torch.float32)
    pad_right = padding_side == "right"
    for i, w in enumerate(chunks):
        length = w.shape[0]
        if length > max_num_samples:
            if pad_right:
                padded[i, :] = w[:max_num_samples]
            else:
                padded[i, :] = w[-max_num_samples:]
        else:
            if pad_right:
                padded[i, :length] = w
            else:
                start_idx = max_num_samples - length
                padded[i, start_idx:] = w
    return padded


def load_audio_features_seq(audio_filepath: str | Path, feature_data: list[tuple[float, float, float, float]],
                            processor, max_num_samples: int):
    wave, sr = torchaudio.load(audio_filepath, backend="soundfile")
    if sr != TARGET_SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE
    if processor is None:
        wave = wave.squeeze(0)
    else:
        wave = wave.squeeze(0).numpy()
    lefts = []
    rights = []
    add_initial_zero_features = False
    for start_feat_1, start_feat_2, num_chunk_samples_1, num_chunk_samples_2 in feature_data:
        if start_feat_1 == 0 and num_chunk_samples_1 == 0 and start_feat_2 == 0:
            add_initial_zero_features = True
            continue
        feat_1 = wave[start_feat_1: start_feat_1 + num_chunk_samples_1]
        feat_2 = wave[start_feat_2: start_feat_2 + num_chunk_samples_2]
        lefts.append(feat_1)
        rights.append(feat_2)
    if processor is None:
        lefts = _create_batched_padded_features(lefts, max_num_samples, padding_side="left")
        rights = _create_batched_padded_features(rights, max_num_samples, padding_side="right")
    else:
        lefts = processor(lefts, sampling_rate=sr, max_length=max_num_samples,
                          padding="max_length", padding_side="left", return_tensors="pt").input_values
        rights = processor(rights, sampling_rate=sr, max_length=max_num_samples,
                           padding="max_length", padding_side="right", return_tensors="pt").input_values
    if add_initial_zero_features:
        lefts = torch.cat([torch.zeros((1, max_num_samples), dtype=torch.float32), lefts], dim=0)
        rights = torch.cat([torch.zeros((1, max_num_samples), dtype=torch.float32), rights], dim=0)
    # stack along new siblings dimension: (seq, 2, wave_len)
    acoustic_features = torch.stack([lefts, rights], dim=1)
    return acoustic_features


def load_from_sentences(sentences: list[dict[str, str | float]], audio_filepath: str | Path,
                        topic_start_tags: list[int] | None, max_len: int, tokenizer: AutoTokenizer,
                        processor: AutoProcessor, chunk_seconds: float, boundary_seconds: float,
                        ):
    # Return: ((seq_semantic, seq_acoustic), seq_labels) or (seq_semantic, seq_acoustic) if no tags
    # with seq_semantic = {"input_ids": ..., "attention_mask": ...}
    # and seq_acoustic = [(audio_path, start_feat_1, start_feat_2, num_chunk_samples_1, num_chunk_samples_2), ...]
    # and seq_labels = Tensor of shape (num_sentences,)
    sent_texts = [s["text"] for s in sentences]
    return_tags = True
    if topic_start_tags is None:
        topic_start_tags = [0 for _ in range(len(sentences))]  # Use dummy tags
        return_tags = False
    text_features = [t for t in load_text_features(sent_texts, topic_start_tags, max_len, tokenizer)]
    audio_features = load_acoustic_feature_times(sentences, chunk_seconds, boundary_seconds)
    # add initial zeroes features to acoustic samples
    # (start_feat_1, start_feat_2, num_chunk_samples_1, num_chunk_samples_2)
    zeros_features = (0, 0, 0, 0)
    audio_features = [zeros_features] + audio_features

    # Check audio seq -> if bad alignment (multiple starts of 0.0), skip whole video
    for n, audio_f in enumerate(audio_features):
        start_feat_1, start_feat_2, num_chunk_samples_1, num_chunk_samples_2 = audio_f
        if start_feat_1 != 0:
            # Check successful, continue with processing
            break
        if n != 0 and num_chunk_samples_1 == 0:
            print(f"--> Skipping {audio_filepath} due to invalid format", flush=True)
            return

    # optionally set this to 1 if loading acoustic samples without initial zero features, and load based on audio
    start_index = 0
    max_num_samples = int(chunk_seconds * TARGET_SAMPLE_RATE)
    for seq_text, seq_labels in text_features:
        num_sentences = seq_text["input_ids"].shape[0]
        end_index = start_index + num_sentences
        seq_audio = audio_features[start_index: end_index]
        seq_audio_features = load_audio_features_seq(
            audio_filepath, seq_audio, processor, max_num_samples=max_num_samples,
        )
        assert seq_audio_features.size(0) == num_sentences, "Mismatch in number of sentences between text and acoustic seq"
        if return_tags:
            full_sample = ((seq_text, seq_audio_features), seq_labels)
        else:
            full_sample = (seq_text, seq_audio_features)
        start_index = end_index
        yield full_sample


def get_topic_idx_for_sentence(s_start: float, s_end: float, topic_intervals: list[tuple[float, float]]) -> int | None:

    def _sentence_belongs_to_topic(_s_start, _s_end, _t_start, _t_end):
        # overlap of at least 50%
        overlap_start = max(_s_start, _t_start)
        overlap_end = min(_s_end, _t_end)
        if overlap_start >= overlap_end:
            return False
        overlap_dur = overlap_end - overlap_start
        sent_dur = _s_end - _s_start
        return overlap_dur / sent_dur >= 0.5

    for t_idx, (t_start, t_end) in enumerate(topic_intervals):
        if _sentence_belongs_to_topic(s_start, s_end, t_start, t_end):
            return t_idx
    return None


def load_tags_from_start_times(sentences: list[dict[str, str | float]], topic_start_times: list[float]
                               ) -> list[int] | None:
    sentences.sort(key=lambda x: x["start"])
    topic_start_times.sort()
    abs_start = min(s["start"] for s in sentences)
    if abs_start < topic_start_times[0]:
        topic_start_times.insert(0, abs_start)  # add initial dummy topic for start tag computation
    abs_end = max([s["end"] for s in sentences])
    topic_intervals = [
        (s, topic_start_times[n+1] if n+1 < len(topic_start_times) else abs_end)
        for n, s in enumerate(topic_start_times)
    ]

    assignments = []
    for sent in sentences:
        topic_idx = get_topic_idx_for_sentence(sent["start"], sent["end"], topic_intervals)
        if topic_idx is None:
            return None
        assignments.append(topic_idx)

    topic_start_tags = []
    for n, t_idx in enumerate(assignments):
        prev_t_idx = assignments[n-1] if n > 0 else None
        if prev_t_idx != t_idx:
            topic_start_tags.append(1)
        else:
            topic_start_tags.append(0)
    return topic_start_tags
