from __future__ import annotations

import warnings

from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from nltk.metrics import segmentation
import segeval


def get_half_segment_size(targets: list[int | float], start_labels: bool = False) -> int:
    num_segments = sum(t > 0 for t in targets) + 1
    if start_labels:
        num_elements = len(targets)
    else:
        num_elements = len(targets) + 1
    half_segment_size = 0.5 * (num_elements / num_segments)
    return max(1, round(half_segment_size))


class Metrics:

    def __init__(self, predictions: list[int], targets: list[int], window_width: int | None = None,
                 name: str | None = None, f1_average: str = "binary", start_labels: bool = False,
                 add_segeval_metrics: bool = False):
        self.predictions = predictions
        self.targets = targets
        self.name = name
        self.f1_average = f1_average
        self.start_labels = start_labels
        self.add_segeval_metrics = add_segeval_metrics
        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
            targets, predictions, pos_label=1, average=f1_average, zero_division=0, labels=[0, 1])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.mcc = matthews_corrcoef(targets, predictions)
        if window_width is None:
            window_width = get_half_segment_size(targets, start_labels)
        if window_width > len(targets):
            # No boundaries in targets, use one full window
            window_width = len(targets)
        self.window_diff = segmentation.windowdiff(predictions, targets, k=window_width, boundary=1)
        mean_segment_length = len(targets) / (sum(targets) + 1)
        self.ghd = segmentation.ghd(ref=targets, hyp=predictions, ins_cost=mean_segment_length,
                                    del_cost=mean_segment_length, shift_cost_coeff=2.0, boundary=1)
        # Compute metrics using segeval
        segeval_metrics = compute_using_segeval(predictions, targets, start_labels)
        self.pk = segeval_metrics["Pk"]
        self.boundary_similarity = segeval_metrics["B"]

    def to_dict(self, include_sequences: bool = False) -> dict[str, float | None]:
        data = {f"f1_{self.f1_average}": self.f1, "precision": self.precision, "recall": self.recall, "mcc": self.mcc,
                "window_diff": self.window_diff, "pk": self.pk,"b": self.boundary_similarity, "ghd": self.ghd,
                "name": self.name}
        if include_sequences:
            data.update({"predictions": self.predictions, "targets": self.targets})
        return data


def merge_metrics(metrics: list[Metrics], round_values: int | None = None, use_weighted_avg: bool = True) -> dict:
    if use_weighted_avg:
        # Normalize in relation to number of changes in video
        normalize_factor = 1 / sum([sum(m.targets) for m in metrics])
        metric_weights = [sum(m.targets) * normalize_factor for m in metrics]
    else:
        metric_weights = [1 / len(metrics) for _ in metrics]
    f1_average = metrics[0].f1_average
    if use_weighted_avg:
        # Weighted averages of classification metrics, use all targets and predictions
        all_targets, all_predictions = [], []
        for metric in metrics:
            all_targets.extend(metric.targets)
            all_predictions.extend(metric.predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, pos_label=1, average=f1_average, zero_division=0, labels=[0, 1])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            mcc = matthews_corrcoef(all_targets, all_predictions)
    else:
        # Compute unweighted averages (every video has equal weight)
        f1 = sum([m.f1 * metric_weights[n] for n, m in enumerate(metrics)])
        precision = sum([m.precision * metric_weights[n] for n, m in enumerate(metrics)])
        recall = sum([m.recall * metric_weights[n] for n, m in enumerate(metrics)])
        mcc = sum([m.mcc * metric_weights[n] for n, m in enumerate(metrics)])

    # Aggregate window metrics either equally (unweighted) or weighted by number of topic changes in video
    overall_metrics = {
        f"f1_{f1_average}": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "window_diff": sum([m.window_diff * metric_weights[n] for n, m in enumerate(metrics)]),
        "pk": sum([m.pk * metric_weights[n] for n, m in enumerate(metrics)]),
        "b": sum([m.boundary_similarity * metric_weights[n] for n, m in enumerate(metrics)]),
        "ghd": sum([m.ghd * metric_weights[n] for n, m in enumerate(metrics)])
    }
    if round_values is not None:
        return {m: round(val, round_values) for m, val in overall_metrics.items()}
    return overall_metrics


def topic_starts_to_masses(label_seq, start_labels: bool) -> list[int]:
    """
    Convert a list of 0/1 topic-start/change labels into a list of segment lengths.
    This function returns the list of segment lengths for segeval.
    """
    if start_labels:
        topic_starts = label_seq
    else:
        topic_starts = [0] + label_seq
    segment_lengths = []
    current_topic_start = 0  # We store the index where the current segment starts
    for i, is_topic_start in enumerate(topic_starts):
        if is_topic_start == 1:
            current_segment_length = i - current_topic_start
            segment_lengths.append(current_segment_length)
            # Start a new topic at i
            current_topic_start = i
    # Add final segment length
    last_segment_length = len(topic_starts) - current_topic_start
    segment_lengths.append(last_segment_length)
    return segment_lengths


def compute_using_segeval(predictions: list[int], targets: list[int], start_labels: bool) -> dict:
    hyp_masses = topic_starts_to_masses(predictions, start_labels)
    ref_masses = topic_starts_to_masses(targets, start_labels)
    pk = segeval.pk(hyp_masses, ref_masses)
    try:
        bs = segeval.boundary_similarity(hyp_masses, ref_masses)
    except ValueError as e:
        bs = 0.0
        print(f"\nError in boundary similarity calculation, using 0.0 as fallback value: "
              f"{e}\nhyp_masses: {hyp_masses}\nref_masses: {ref_masses}\n", flush=True)
    return {"Pk": float(pk), "B": float(bs)}
