import argparse
import base64
import os
import time

from common import SPLITS, audio_to_wav_bytes, build_caption_question
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-audio-preview")
    parser.add_argument("--output_dir", type=str, default="text/captions/gpt4omini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_detailed_caption", action="store_true")
    parser.add_argument("--queries_only", action="store_true")

    return parser.parse_args()


def prepare_inputs(item, use_detailed_caption=False):
    audio = item["audio"]
    wav_bytes = audio_to_wav_bytes(audio["array"], audio["sampling_rate"])
    encoded_audio = base64.b64encode(wav_bytes).decode("ascii")
    question = build_caption_question(item, use_detailed_caption)
    content = []
    content.append(
        {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": "wav"}}
    )
    content.append({"type": "text", "text": question})

    return content


def extract_captions(client, model_name, items, items_dir, use_detailed_caption=False):
    os.makedirs(items_dir, exist_ok=True)

    for idx, item in enumerate(tqdm(items, desc="Preparing inputs")):
        content = prepare_inputs(item, use_detailed_caption)
        messages = [{"role": "user", "content": content}]
        max_retries = 20
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model_name, modalities=["text"], messages=messages
                )
                response_text = completion.choices[0].message.content.strip()
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
                    print("Failed messages:\n", content)
                    response_text = ""
        with open(os.path.join(items_dir, f"{idx:05d}.txt"), "w") as f:
            f.write(response_text)


def main(args):
    set_seed(args.seed)

    # Load dataset
    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for split in SPLITS:
        print(f"\nProcessing {split} split...")

        # Extract captions for queries
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
