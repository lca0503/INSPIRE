import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def mean_pooling(model_output, attention_mask):
    """Mean pooling."""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def extract_queries_rep(
    model,
    tokenizer,
    max_length,
    queries,
    queries_representations_dir,
):
    """Extract queries representations."""
    os.makedirs(queries_representations_dir, exist_ok=True)
    for idx, query in enumerate(tqdm(queries, desc="Extract queries representations")):
        query_instruction = query["instruction"]
        query_transcription = query["text"]
        query_speaker = query["speaker"]
        query_speaking_style = query["speaking_style"]
        query_environmental_sound = query["environmental_sound"]

        query_caption = f"Speaker: {query_speaker}, Speaking Style: {query_speaking_style}, Environmental Sound: {query_environmental_sound}"

        input_text = f"Instruct: {query_instruction}\nQuery: Transcription: {query_transcription} Caption: {query_caption}"

        inputs = tokenizer(
            [input_text],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = embeddings.squeeze(0).detach().cpu().float().numpy()

        np.save(os.path.join(queries_representations_dir, f"{idx:05d}.npy"), embeddings)


def extract_documents_rep(
    model, tokenizer, max_length, documents, documents_representations_dir
):
    """Extract documents representations."""
    os.makedirs(documents_representations_dir, exist_ok=True)
    for document in tqdm(documents, desc="Extract documents representations"):
        audio_id = document["audio_id"]
        document_transcription = document["text"]
        document_speaker = document["speaker"]
        document_speaking_style = document["speaking_style"]
        document_environmental_sound = document["environmental_sound"]

        document_caption = f"Speaker: {document_speaker}, Speaking Style: {document_speaking_style}, Environmental Sound: {document_environmental_sound}"

        input_text = (
            f"Transcription: {document_transcription} Caption: {document_caption}"
        )
        inputs = tokenizer(
            [input_text],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = embeddings.squeeze(0).detach().cpu().float().numpy()

        np.save(
            os.path.join(documents_representations_dir, f"{audio_id}.npy"), embeddings
        )


def main(args):
    """Main function."""
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name, device_map="auto", trust_remote_code=True
    )
    model.eval()

    splits = ["DailyTalk", "Expresso", "VCTK", "Synthetic"]

    for split in splits:
        queries_representations_dir = os.path.join(args.output_dir, "queries", split)
        extract_queries_rep(
            model,
            tokenizer,
            args.max_length,
            queries[split],
            queries_representations_dir,
        )
        documents_representations_dir = os.path.join(
            args.output_dir, "documents", split
        )
        extract_documents_rep(
            model,
            tokenizer,
            args.max_length,
            documents[split],
            documents_representations_dir,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
