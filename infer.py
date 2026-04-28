from argparse import ArgumentParser
import os

from mts.inferrer import Inferrer
from mts.utils import load_json_data, save_json_data

"""
Expected input format:
Directory containing JSON files, each with:
{
    "audio_filepath": "/path/to/audio/file.wav",
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


def infer_for_sample(data: dict[str, str | list[dict]], inferrer: Inferrer) -> tuple[list[int], list[float]] | None:
    audio_filepath = data.get("audio_filepath")
    sentences = data.get("sentences")
    if not audio_filepath or not sentences:
        return None
    predictions = inferrer(sentences, audio_filepath)
    topic_start_times = [sentences[i]["start"] for i, tag in enumerate(predictions) if tag == 1]
    return predictions, topic_start_times


def main():
    parser = ArgumentParser(description="Infer topic segments using a trained model.")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="Path to input data for inference.")
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoints/multi-seg",
                        help="Path to model checkpoint directory.")
    parser.add_argument("-o", "--output-file", type=str, default=None, help="Path to output json file.")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise ValueError(f"Data path not found: {args.data}")
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint}")

    inferrer = Inferrer(checkpoint=args.checkpoint)
    filepaths = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(".json")]
    print(f"Predicting topic start times for {len(filepaths)} input samples in directory {args.data}.")
    print("---------------------------------------------")
    results = dict()
    for fp in filepaths:
        file_data = load_json_data(fp)
        predictions = infer_for_sample(file_data, inferrer)
        if predictions is None:
            print(f"- Skipping file {fp} due to invalid format.")
        else:
            tags, topic_start_times = predictions
            results[fp] = {"tags": tags, "topic_start_times": topic_start_times}
            print(f"- {fp}:\n  {topic_start_times}")
        print("---------------------------------------------")
    if args.output_file:
        save_json_data(results, args.output_file)


if __name__ == "__main__":
    main()
