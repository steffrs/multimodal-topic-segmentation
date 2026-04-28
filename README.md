# multimodal-topic-segmentation
Repository for the paper "Reading Between the Waves: Robust Topic Segmentation Using Inter-Sentence Audio Features" (ICASSP 2026).

The repository contains code for inference and evaluation of the topic segmentation method proposed in the paper.
This includes model checkpoints ([`checkpoints/`](./checkpoints)) 
and scripts for running inference ([`infer.py`](infer.py)) 
and evaluation ([`evaluate.py`](evaluate.py)).


## Setup
Create a new environment (python=3.12) and install the required packages:
```bash
pip install -r requirements.txt
```


## Inference
To run inference on a set of samples (transcript + audiofile), use the following command:
```bash
python infer.py -d /path/to/directory/with/samples
```
with the following command line arguments:
- `-d` or `--data`: Path to the directory containing the json sample files (see the format below).
- `-c` or `--checkpoint`: Path to the model checkpoint to use for inference 
  (default: [`checkpoints/multi-seg`](./checkpoints/multi-seg)).
- `-o` or `--output-file`: Optional path of the output file where the predicted topic starts will be saved 
  (default: `None`).

#### Input Format
The input samples should be in json format, with the following structure:
```json
{
  "audio_filepath": "/path/to/audio/file.wav",
  "sentences":
    [
      {
        "text": "Transcribed sentence.",
        "start": 0.0,
        "end": 5.62
      },
      ...
    ]
}
```


## Evaluation
To evaluate the models using labeled data, use the following command:
```bash
python evaluate.py -d /path/to/directory/with/samples
```
Here, the same command line options as for `infer.py` are available, with the addition of:
- `--bootstrap`: Flag to enable bootstrapping of samples and calculation of stddev for each metric (default: `False`).

The expected format is the same as for `infer.py`, with the addition of the target topic starts.
These can be provided in either of the following two formats (top-level json fields):
- `targets`: List of topic start tags (1=start of new topic, 0=no topic change), e.g.: `[0, 0, 1, 0, ...]`.
- `topic_start_times`: List of target topic start times in seconds, e.g.: `[0.0, 43.2, 128.6, 202.1, ...]`.


## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{freisinger2026readingwavesrobusttopic,
      title={Reading Between the Waves: Robust Topic Segmentation Using Inter-Sentence Audio Features}, 
      author={Steffen Freisinger and Philipp Seeberger and Tobias Bocklet and Korbinian Riedhammer},
      year={2026},
      eprint={2602.06647},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06647}, 
}
```
