import argparse
import os
import time

from common import SPLITS, audio_to_wav_bytes, build_caption_question
from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm
from transformers import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--output_dir", type=str, default="text/captions/gemini3flash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_detailed_caption", action="store_true")
    parser.add_argument("--queries_only", action="store_true")

    return parser.parse_args()


def prepare_inputs(item, use_detailed_caption=False):
    item_audio = item["audio"]
    audio = audio_to_wav_bytes(item_audio["array"], item_audio["sampling_rate"])
    question = build_caption_question(item, use_detailed_caption)
    contents = [
        types.Part.from_bytes(data=audio, mime_type="audio/wav"),
        question,
    ]

    return contents


def extract_captions(client, model_name, items, items_dir, use_detailed_caption=False):
    os.makedirs(items_dir, exist_ok=True)

    for idx, item in enumerate(tqdm(items, desc="Preparing inputs")):
        contents = prepare_inputs(item, use_detailed_caption)
        max_retries = 20
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                response_text = response.text.strip()
                if response_text == "":
                    raise ValueError("Error: Empty response received")
                break
            except Exception as error:  # noqa: BLE001 - retry transient API failures
                if attempt < max_retries - 1:
                    print(
                        f"Retrying ({attempt + 1}/{max_retries}) due to error: {error}"
                    )
                    time.sleep(retry_delay)
                else:
                    print(
                        f"Failed after {max_retries} attempts for sample {item.get('audio_id', '')}: {error}"
                    )
                    print("Failed messages:\n", contents)
                    response_text = ""
        with open(os.path.join(items_dir, f"{idx:05d}.txt"), "w") as f:
            f.write(response_text)


def main(args):
    set_seed(args.seed)

    # Load dataset
    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    for split in SPLITS:
        print(f"\nProcessing {split} split...")

        queries_dir = os.path.join(args.output_dir, "queries", split)
        extract_captions(
            client,
            args.model_name,
            queries[split],
            queries_dir,
            args.use_detailed_caption,
        )

        if not args.queries_only:
            documents_dir = os.path.join(args.output_dir, "documents", split)
            extract_captions(
                client,
                args.model_name,
                documents[split],
                documents_dir,
                args.use_detailed_caption,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
