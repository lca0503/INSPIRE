import argparse
import os
import shutil
import warnings

import librosa
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, default="facebook/hubert-large-ll60k")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def extract_rep(model, feature_extractor, layer, target_sr, items, items_dir, device):
    """Extract representations."""
    os.makedirs(items_dir, exist_ok=True)
    for idx, item in enumerate(tqdm(items, desc="Extract representations")):
        audio_id = item["audio_id"]

        previous_output_path = os.path.join(items_dir, f"{audio_id}.npy")
        output_path = os.path.join(items_dir, f"{idx:05d}.npy")
        if os.path.exists(previous_output_path) and previous_output_path != output_path:
            shutil.copy(previous_output_path, output_path)
            continue

        array = item["audio"]["array"]
        orig_sr = item["audio"]["sampling_rate"]

        if orig_sr != target_sr:
            array = librosa.resample(array, orig_sr=orig_sr, target_sr=target_sr)

        inputs = feature_extractor(
            array, padding=False, sampling_rate=target_sr, return_tensors="pt"
        )
        inputs = {k: v.to(dtype=torch.float32).to(device) for k, v in inputs.items()}

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*Support for mismatched key_padding_mask and attn_mask is deprecated.*",
            )
            outputs = model(**inputs)

        pooled_output = torch.mean(outputs.hidden_states[layer], dim=1)

        rep = pooled_output.squeeze(0).detach().cpu().float().numpy()

        np.save(output_path, rep)


def main(args):
    """Main function."""
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dtype=torch.float32).to(device)
    model.eval()

    splits = ["DailyTalk", "Expresso", "VCTK", "Synthetic"]

    for split in splits:
        queries_dir = os.path.join(args.output_dir, "queries", split)
        extract_rep(
            model,
            feature_extractor,
            args.layer,
            args.sampling_rate,
            queries[split],
            queries_dir,
            device,
        )
        documents_dir = os.path.join(args.output_dir, "documents", split)
        extract_rep(
            model,
            feature_extractor,
            args.layer,
            args.sampling_rate,
            documents[split],
            documents_dir,
            device,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
