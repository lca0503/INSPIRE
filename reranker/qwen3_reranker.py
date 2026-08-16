import argparse
import os

import torch
from common import (
    SPLITS,
    load_text_files,
    load_top_retrieved_docs,
    write_rerank_entry,
)
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rerank documents using Qwen3-Reranker-8B (text from transcription and caption)"
    )
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--transcriptions_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100]
    )
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def format_instruction(instruction, query, doc):
    """Format (instruction, query, doc) for Qwen3-Reranker."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def process_inputs(pairs, tokenizer, prefix_tokens, suffix_tokens, max_length):
    """Tokenize pairs and add prefix/suffix."""
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(
        inputs, padding=True, return_tensors="pt", max_length=max_length
    )
    return inputs


@torch.no_grad()
def compute_logits(model, inputs, token_true_id, token_false_id):
    """Compute relevance scores (yes/no log-softmax)."""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def rerank(
    model,
    tokenizer,
    query_text,
    doc_ids,
    doc_id_to_text,
    instruction,
    prefix_tokens,
    suffix_tokens,
    max_length,
    token_true_id,
    token_false_id,
    batch_size,
):
    """Rerank documents by scoring (query, doc) pairs in batches."""
    pairs = [
        format_instruction(instruction, query_text, doc_id_to_text[doc_id])
        for doc_id in doc_ids
    ]
    rerank_scores = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        batch_doc_ids = doc_ids[start : start + batch_size]
        inputs = process_inputs(
            batch_pairs, tokenizer, prefix_tokens, suffix_tokens, max_length
        )
        device = next(model.parameters()).device
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        scores = compute_logits(model, inputs, token_true_id, token_false_id)
        for doc_id, score in zip(batch_doc_ids, scores):
            rerank_scores.append((doc_id, score))
    rerank_scores.sort(key=lambda x: x[1], reverse=True)
    return rerank_scores


def main(args):
    """Main function."""
    set_seed(args.seed)

    print(f"Loading dataset: {args.dataset_name}")
    queries_ds = load_dataset(args.dataset_name, "query")
    documents_ds = load_dataset(args.dataset_name, "document")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    ).eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192

    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    k_values = sorted(args.k_values)

    for split in SPLITS:
        print(f"Find {len(queries_ds[split])} queries in {split} split")
        print(f"Find {len(documents_ds[split])} documents in {split} split")

        queries_transcriptions_dir = os.path.join(
            args.transcriptions_dir, "queries", split
        )
        queries_captions_dir = os.path.join(args.captions_dir, "queries", split)
        documents_transcriptions_dir = os.path.join(
            args.transcriptions_dir, "documents", split
        )
        documents_captions_dir = os.path.join(args.captions_dir, "documents", split)

        query_transcriptions = load_text_files(queries_transcriptions_dir)
        query_captions = load_text_files(queries_captions_dir)
        doc_transcriptions = load_text_files(documents_transcriptions_dir)
        doc_captions = load_text_files(documents_captions_dir)

        top_retrieved_docs = load_top_retrieved_docs(
            os.path.join(args.input_dir, split, "score.jsonl")
        )
        output_file = os.path.join(args.output_dir, split, "score.jsonl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            for idx, query in enumerate(
                tqdm(queries_ds[split], desc="Reranking queries")
            ):
                query_audio_id = query["audio_id"]
                query_transcription = query_transcriptions[query_audio_id]
                query_caption = query_captions[f"{idx:05d}"]
                if args.no_instruction:
                    query_text = (
                        f"Transcription: {query_transcription} Caption: {query_caption}"
                    )
                    instruction = ""
                else:
                    query_text = (
                        f"Transcription: {query_transcription} Caption: {query_caption}"
                    )
                    instruction = query["instruction"]

                top_retrieved_docs_ids = top_retrieved_docs[f"{idx:05d}"]
                doc_id_to_text = {}
                for doc_id in top_retrieved_docs_ids:
                    doc = documents_ds[split][int(doc_id)]
                    doc_audio_id = doc["audio_id"]
                    doc_transcription = doc_transcriptions[doc_audio_id]
                    doc_caption = doc_captions[doc_id]
                    doc_id_to_text[doc_id] = (
                        f"Transcription: {doc_transcription} Caption: {doc_caption}"
                    )

                rerank_scores = rerank(
                    model,
                    tokenizer,
                    query_text,
                    top_retrieved_docs_ids,
                    doc_id_to_text,
                    instruction,
                    prefix_tokens,
                    suffix_tokens,
                    max_length,
                    token_true_id,
                    token_false_id,
                    args.batch_size,
                )
                reranked_docs_ids = [doc_id for doc_id, _ in rerank_scores]
                write_rerank_entry(f, f"{idx:05d}", query, reranked_docs_ids, k_values)


if __name__ == "__main__":
    args = parse_args()
    main(args)
