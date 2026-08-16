import argparse
import os

from common import SPLITS, build_caption_question, resample_audio
from datasets import load_dataset
from tqdm import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_detailed_caption", action="store_true")
    parser.add_argument("--queries_only", action="store_true")

    return parser.parse_args()


def extract_captions(
    model,
    processor,
    items,
    items_dir,
    max_new_tokens,
    use_detailed_caption=False,
):
    """Extract captions for items using transformers."""
    os.makedirs(items_dir, exist_ok=True)
    target_sampling_rate = processor.feature_extractor.sampling_rate

    for idx, item in enumerate(tqdm(items, desc="Extract captions")):
        audio = resample_audio(item, target_sampling_rate)

        question = build_caption_question(item, use_detailed_caption)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        output_path = os.path.join(items_dir, f"{idx:05d}.txt")
        with open(output_path, "w") as file:
            file.write(decoded_outputs[0])


def main(args):
    """Extract captions for INSPIRE."""
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_name, device_map="auto"
    )
    model.eval()

    for split in SPLITS:
        print(f"\nProcessing {split} split...")
        queries_dir = os.path.join(args.output_dir, "queries", split)
        extract_captions(
            model,
            processor,
            queries[split],
            queries_dir,
            args.max_new_tokens,
            args.use_detailed_caption,
        )
        if not args.queries_only:
            documents_dir = os.path.join(args.output_dir, "documents", split)
            extract_captions(
                model,
                processor,
                documents[split],
                documents_dir,
                args.max_new_tokens,
                args.use_detailed_caption,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
