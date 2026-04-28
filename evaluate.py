from argparse import ArgumentParser
import os
import random
from collections import defaultdict

from tqdm import tqdm
import numpy

from mts.inferrer import Inferrer
from mts.utils import load_json_data, save_json_data
from infer import infer_for_sample
from mts.metrics import Metrics, merge_metrics
from mts.load import load_tags_from_start_times


"""
Expected input format:
Directory containing JSON files, each with:
{
    "audio_filepath": "/path/to/audio/file.wav",
    "targets": [0, 0, 1, 0, ...],  # List of topic start tags (1=start of new topic, 0=no topic change)
    "topic_start_times": [0.0, 43.2, 128.6, 202.1, ...],  # Alternative for topic start tags: List of topic start times in seconds
    "sentences":
        [
            {
                "text": "Transcription of the sentence.",
                "start": float (in seconds),
                "end": float (in seconds)
            },
            ...
        ],
}
"""


METRICS_OF_INTEREST = ["f1_binary", "precision", "recall", "pk", "b"]


def aggregate_and_print_metrics(metrics: list[Metrics]) -> dict:
    m = merge_metrics(metrics, round_values=4, use_weighted_avg=True)
    print("---------------------------------------------")
    print("Average metrics (weighted):")
    for k, v in m.items():
        if k in METRICS_OF_INTEREST:
            print(f"- {k}: {v}")
    print("---------------------------------------------")
    return m


def aggregate_and_print_bootstrapped_metrics(metrics: list[dict], initial_mark: str | None = None
                                             ) -> dict[str, tuple[float, float]]:
    # Calculate avg metrics with std deviation
    initial_mark = "" if initial_mark is None else initial_mark
    metric_2_values = defaultdict(list)
    for m in metrics:
        for k, v in m.items():
            if k in METRICS_OF_INTEREST:
                metric_2_values[k].append(v)
    bootstrapped_metrics = {}
    print("---------------------------------------------")
    print("Bootstrapped weighted average metrics:")
    for m, vs in metric_2_values.items():
        avg = numpy.average(vs)
        std_dev = numpy.std(vs)
        print(f"{initial_mark}- {m}: {round(avg, 4)} +/- {round(std_dev, 4)}")
        bootstrapped_metrics[m] = (avg, std_dev)
    print("---------------------------------------------")
    return bootstrapped_metrics


def load_and_verify_sample(input_file) -> tuple[dict | None, list | None]:
    file_data = load_json_data(input_file)
    sentences = file_data.get("sentences")
    if not sentences:
        print(f"- Skipping file {input_file} due to missing sentences.")
        return None, None
    targets = file_data.get("targets")
    if targets is None:
        topic_start_times = file_data.get("topic_start_times")
        if topic_start_times is None:
            print(f"- Skipping file {input_file} due to missing any of: `targets` or `topic_start_times`.")
            return None, None
        targets = load_tags_from_start_times(sentences, topic_start_times)
        if targets is None:
            print(f"- Skipping file {input_file}, failed to assign topic change tags based on topic start times.")
    if not len(targets) == len(sentences):
        print(f"- Skipping file {input_file} due to missmatch of num sentences and num targets.")
        return None, None
    return file_data, targets


def main():
    parser = ArgumentParser(description="Evaluate topic segmentation on a labeled dataset.")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="Path to input data for evaluation.")
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoints/multi-seg",
                        help="Path to model checkpoint directory.")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap samples 100 times.")
    parser.add_argument("-o", "--output-file", type=str, default=None, help="Path to output json file.")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise ValueError(f"Data path not found: {args.data}")
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint}")

    inferrer = Inferrer(checkpoint=args.checkpoint)
    filepaths = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(".json")]
    filepaths.sort()

    all_metrics = []
    results = dict()
    print("---------------------------------------------")
    print(f"Starting evaluation of {len(filepaths)} samples in directory {args.data}.")
    for fp in tqdm(filepaths, desc="Evaluating samples", unit="sample"):
        file_data, targets = load_and_verify_sample(fp)
        if not file_data:
            continue
        res = infer_for_sample(file_data, inferrer)
        if not res:
            print(f"- Skipping file {fp} due to invalid format.")
            continue
        predictions, topic_start_times = res
        # first sentence is not considered a potential topic start (always the same tag), thus we skip it
        metrics = Metrics(predictions[1:], targets[1:], name=fp, start_labels=False, add_segeval_metrics=True)
        results[fp] = {"tags": predictions, "topic_start_times": topic_start_times, "metrics": metrics.to_dict()}
        all_metrics.append(metrics)

    print(f"Evaluated {len(all_metrics)} samples, skipped {len(filepaths) - len(all_metrics)} samples.")
    if not all_metrics:
        return

    if args.bootstrap:
        # Bootstrap metrics 100 times
        all_avg_metrics = []
        num_metrics = len(all_metrics)
        for n in range(100):
            random.seed(n)  # Set seed to always use the same bootstrap at iteration n
            bootstrapped_metrics = random.choices(all_metrics, k=num_metrics)
            avg_metrics = merge_metrics(bootstrapped_metrics, use_weighted_avg=True)
            all_avg_metrics.append(avg_metrics)
        metrics_to_save = aggregate_and_print_bootstrapped_metrics(all_avg_metrics)
    else:
        metrics_to_save = aggregate_and_print_metrics(all_metrics)

    if args.output_file:
        out = {"metrics": metrics_to_save, "results": results}
        save_json_data(out, args.output_file)


if __name__ == "__main__":
    main()
