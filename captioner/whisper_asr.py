import argparse
import os

from common import SPLITS, resample_audio
from datasets import load_dataset
from faster_whisper import WhisperModel
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="float16")

    return parser.parse_args()


def transcribe_speech(model, items, output_dir):
    """Transcribe speech using Whisper."""
    os.makedirs(output_dir, exist_ok=True)
    for item in tqdm(items, desc="Transcribing"):
        audio_id = item["audio_id"]
        output_file = os.path.join(output_dir, f"{audio_id}.txt")
        if os.path.exists(output_file):
            continue

        audio = resample_audio(item, 16000)
        segments, _ = model.transcribe(audio)
        transcription = " ".join(segment.text for segment in segments).strip()

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcription)


def main(args):
    """Transcribe speech using Whisper."""
    print("Loading Whisper model: large-v3")
    model = WhisperModel("large-v3", device=args.device, compute_type=args.compute_type)

    print(f"Loading dataset: {args.dataset_name}")
    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    for split in SPLITS:
        print(f"\nProcessing {split} split...")

        queries_dir = os.path.join(args.output_dir, "queries", split)
        transcribe_speech(model, queries[split], queries_dir)

        documents_dir = os.path.join(args.output_dir, "documents", split)
        transcribe_speech(model, documents[split], documents_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
