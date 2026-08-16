import argparse
import os

import numpy as np
from common import SPLITS, load_text_files
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--transcriptions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--queries_only", action="store_true")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_embedding(text, client, model_name):
    text = text.replace("\n", " ")
    embeddings = (
        client.embeddings.create(input=[text], model=model_name).data[0].embedding
    )
    return np.array(embeddings)


def extract_queries_rep(
    client,
    model_name,
    queries,
    queries_transcriptions_dir,
    queries_captions_dir,
    queries_representations_dir,
    no_instruction,
):
    transcriptions = load_text_files(queries_transcriptions_dir)
    captions = load_text_files(queries_captions_dir)
    os.makedirs(queries_representations_dir, exist_ok=True)
    for idx, query in enumerate(tqdm(queries, desc="Extract queries representations")):
        audio_id = query["audio_id"]
        query_instruction = query["instruction"]
        query_transcription = transcriptions[audio_id]
        query_caption = captions[f"{idx:05d}"]

        if no_instruction:
            input_text = (
                f"Transcription: {query_transcription} Caption: {query_caption}"
            )
        else:
            input_text = f"Instruct: {query_instruction}\nQuery: Transcription: {query_transcription} Caption: {query_caption}"

        embeddings = get_embedding(input_text, client, model_name)

        np.save(os.path.join(queries_representations_dir, f"{idx:05d}.npy"), embeddings)


def extract_documents_rep(
    client,
    model_name,
    documents,
    documents_transcriptions_dir,
    documents_captions_dir,
    documents_representations_dir,
):
    transcriptions = load_text_files(documents_transcriptions_dir)
    captions = load_text_files(documents_captions_dir)
    os.makedirs(documents_representations_dir, exist_ok=True)
    for idx, document in enumerate(
        tqdm(documents, desc="Extract documents representations")
    ):
        audio_id = document["audio_id"]
        document_transcription = transcriptions[audio_id]
        document_caption = captions[f"{idx:05d}"]

        input_text = (
            f"Transcription: {document_transcription} Caption: {document_caption}"
        )
        embeddings = get_embedding(input_text, client, model_name)

        np.save(
            os.path.join(documents_representations_dir, f"{idx:05d}.npy"), embeddings
        )


def main(args):
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for split in SPLITS:
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
        extract_queries_rep(
            client,
            args.model_name,
            queries[split],
            queries_transcriptions_dir,
            queries_captions_dir,
            queries_representations_dir,
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
        extract_documents_rep(
            client,
            args.model_name,
            documents[split],
            documents_transcriptions_dir,
            documents_captions_dir,
            documents_representations_dir,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
