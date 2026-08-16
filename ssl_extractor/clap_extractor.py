import argparse
import os

import librosa
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    ClapAudioModelWithProjection,
    ClapProcessor,
    ClapTextModelWithProjection,
    set_seed,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, default="laion/clap-htsat-fused")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "audio"],
        required=True,
        help="Extract text or audio representations",
    )
    parser.add_argument(
        "--captions_dir",
        type=str,
        default=None,
        help="Directory containing caption files",
    )
    parser.add_argument(
        "--transcriptions_dir",
        type=str,
        default=None,
        help="Directory containing transcription files",
    )
    parser.add_argument(
        "--queries_only",
        action="store_true",
        help="Extract only query representations",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        help="Extract text representations without instruction",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_transcriptions(transcriptions_dir):
    """Load transcriptions."""
    transcriptions = {}
    for file in os.listdir(transcriptions_dir):
        with open(os.path.join(transcriptions_dir, file), "r") as f:
            transcriptions[file.split(".")[0]] = f.read()
    return transcriptions


def load_captions(captions_dir):
    """Load captions."""
    captions = {}
    for file in os.listdir(captions_dir):
        with open(os.path.join(captions_dir, file), "r") as f:
            captions[file.split(".")[0]] = f.read()
    return captions


def extract_text_rep(
    model,
    tokenizer,
    items,
    transcriptions_dir,
    captions_dir,
    representations_dir,
    device,
    no_instruction=False,
):
    """Extract text representations."""
    transcriptions = load_transcriptions(transcriptions_dir)
    captions = load_captions(captions_dir)
    os.makedirs(representations_dir, exist_ok=True)
    for idx, item in enumerate(tqdm(items, desc="Extract text representations")):
        audio_id = item["audio_id"]
        item_transcription = transcriptions[audio_id]
        item_caption = captions[audio_id]
        if no_instruction or item.get("instruction", "") == "":
            text = f"Transcription: {item_transcription} Caption: {item_caption}"
        else:
            item_instruction = item["instruction"]
            text = f"Instruct: {item_instruction}\nQuery: Transcription: {item_transcription} Caption: {item_caption}"

        inputs = tokenizer(
            [text], max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds

        rep = text_embeds.squeeze(0).detach().cpu().float().numpy()

        np.save(os.path.join(representations_dir, f"{idx:05d}.npy"), rep)


def extract_audio_rep(model, processor, items, representations_dir, device):
    """Extract audio representations."""
    os.makedirs(representations_dir, exist_ok=True)
    for idx, item in enumerate(tqdm(items, desc="Extract audio representations")):
        array = item["audio"]["array"]
        orig_sr = item["audio"]["sampling_rate"]

        if orig_sr != 48000:
            array = librosa.resample(array, orig_sr=orig_sr, target_sr=48000)

        inputs = processor(audio=array, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            audio_embeds = outputs.audio_embeds

        rep = audio_embeds.squeeze(0).detach().cpu().float().numpy()

        np.save(os.path.join(representations_dir, f"{idx:05d}.npy"), rep)


def main(args):
    """Main function."""
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "text":
        model = ClapTextModelWithProjection.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = model.to(device)
        model.eval()

        splits = ["DailyTalk", "Expresso", "VCTK", "Synthetic"]

        for split in splits:
            queries_transcriptions_dir = os.path.join(
                args.transcriptions_dir, "queries", split
            )
            queries_captions_dir = os.path.join(args.captions_dir, "queries", split)
            if args.no_instruction:
                queries_representations_dir = os.path.join(
                    args.output_dir, "queries_no_inst", split
                )
            else:
                queries_representations_dir = os.path.join(
                    args.output_dir, "queries", split
                )
            extract_text_rep(
                model,
                tokenizer,
                queries[split],
                queries_transcriptions_dir,
                queries_captions_dir,
                queries_representations_dir,
                device,
                args.no_instruction,
            )
            if args.queries_only:
                continue
            documents_transcriptions_dir = os.path.join(
                args.transcriptions_dir, "documents", split
            )
            documents_captions_dir = os.path.join(args.captions_dir, "documents", split)
            documents_representations_dir = os.path.join(
                args.output_dir, "documents", split
            )
            extract_text_rep(
                model,
                tokenizer,
                documents[split],
                documents_transcriptions_dir,
                documents_captions_dir,
                documents_representations_dir,
                device,
                args.no_instruction,
            )

    elif args.mode == "audio":
        model = ClapAudioModelWithProjection.from_pretrained(args.model_name)
        processor = ClapProcessor.from_pretrained(args.model_name)
        model = model.to(device)
        model.eval()

        splits = ["DailyTalk", "Expresso", "VCTK", "Synthetic"]

        for split in splits:
            queries_dir = os.path.join(args.output_dir, "queries", split)
            extract_audio_rep(model, processor, queries[split], queries_dir, device)
            if args.queries_only:
                continue
            documents_dir = os.path.join(args.output_dir, "documents", split)
            extract_audio_rep(model, processor, documents[split], documents_dir, device)


if __name__ == "__main__":
    args = parse_args()
    main(args)
