import argparse
import os

import numpy as np
import torch
from common import resample_audio
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, VoxtralForConditionalGeneration, set_seed

SPLITS = ("DailyTalk", "Expresso", "VCTK", "Synthetic")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--queries_only", action="store_true")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--mean_pooling", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def extract_queries_rep(
    model, processor, queries, queries_dir, no_instruction, mean_pooling
):
    """Extract queries representations."""
    os.makedirs(queries_dir, exist_ok=True)
    for idx, query in enumerate(tqdm(queries, desc="Extract queries representations")):
        instruction = query["instruction"]
        audio = resample_audio(query, processor.feature_extractor.sampling_rate)

        if no_instruction:
            user_content = [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Summarize above speech in one word:"},
            ]
        elif mean_pooling:
            user_content = [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio": audio},
            ]
        else:
            user_content = [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio": audio},
                {
                    "type": "text",
                    "text": "Summarize above sentence and speech in one word:",
                },
            ]

        conversation = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True, return_dict=True)

        if mean_pooling:
            rep = (
                output.hidden_states[-1]
                .mean(dim=1)
                .squeeze(0)
                .detach()
                .cpu()
                .float()
                .numpy()
            )
        else:
            rep = (
                output.hidden_states[-1][:, -1, :]
                .squeeze(0)
                .detach()
                .cpu()
                .float()
                .numpy()
            )

        np.save(os.path.join(queries_dir, f"{idx:05d}.npy"), rep)


def extract_documents_rep(model, processor, documents, documents_dir, mean_pooling):
    """Extract documents representations."""
    os.makedirs(documents_dir, exist_ok=True)
    for document in tqdm(documents, desc="Extract documents representations"):
        audio = resample_audio(document, processor.feature_extractor.sampling_rate)

        if mean_pooling:
            user_content = [
                {"type": "audio", "audio": audio},
            ]
        else:
            user_content = [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Summarize above speech in one word:"},
            ]

        conversation = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True, return_dict=True)

        if mean_pooling:
            rep = (
                output.hidden_states[-1]
                .mean(dim=1)
                .squeeze(0)
                .detach()
                .cpu()
                .float()
                .numpy()
            )
        else:
            rep = (
                output.hidden_states[-1][:, -1, :]
                .squeeze(0)
                .detach()
                .cpu()
                .float()
                .numpy()
            )

        np.save(os.path.join(documents_dir, f"{document['audio_id']}.npy"), rep)


def main(args):
    """Extract representations for INSPIRE."""
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    for split in SPLITS:
        if args.no_instruction:
            queries_dir = os.path.join(args.output_dir, "queries_no_inst", split)
        else:
            queries_dir = os.path.join(args.output_dir, "queries", split)
        extract_queries_rep(
            model,
            processor,
            queries[split],
            queries_dir,
            args.no_instruction,
            args.mean_pooling,
        )
        if not args.queries_only:
            documents_dir = os.path.join(args.output_dir, "documents", split)
            extract_documents_rep(
                model, processor, documents[split], documents_dir, args.mean_pooling
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
