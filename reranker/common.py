"""Shared, model-independent reranker utilities."""

import json
import os

import librosa
import numpy as np

SPLITS = ("DailyTalk", "Expresso", "VCTK", "Synthetic")


def resample_audio(item, target_sampling_rate):
    """Return an item's audio array resampled to the model sampling rate."""
    audio = item["audio"]
    return librosa.resample(
        audio["array"],
        orig_sr=audio["sampling_rate"],
        target_sr=target_sampling_rate,
    )


def load_top_retrieved_docs(input_file):
    """Load the top retrieved document IDs for each query."""
    top_retrieved_docs = {}
    with open(input_file, "r") as file:
        for line in file:
            data = json.loads(line)
            top_retrieved_docs[data["id"]] = data["top-100-retrieved-docs"]
    return top_retrieved_docs


def load_text_files(directory):
    """Load text files into a dictionary keyed by filename stem."""
    texts = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as file:
            texts[filename.split(".")[0]] = file.read()
    return texts


def calculate_recall_at_k(retrieved_ids, positive_docs, k):
    """Calculate Recall@k."""
    if len(positive_docs) == 0:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    return len(top_k_retrieved & positive_docs) / len(positive_docs)


def calculate_ndcg_at_k(retrieved_ids, positive_docs, k):
    """Calculate NDCG@k."""
    if len(positive_docs) == 0:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in positive_docs
    )
    num_relevant = min(len(positive_docs), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, num_relevant + 1))
    return dcg / idcg if idcg else 0.0


def write_rerank_entry(file, query_id, query, reranked_docs_ids, k_values):
    """Calculate metrics and write one reranker JSONL entry."""
    positive_docs = set(query["positive_documents"])
    recalls = {
        f"recall@{k}": float(calculate_recall_at_k(reranked_docs_ids, positive_docs, k))
        for k in k_values
    }
    ndcgs = {
        f"ndcg@{k}": float(calculate_ndcg_at_k(reranked_docs_ids, positive_docs, k))
        for k in k_values
    }
    result_entry = {
        "id": query_id,
        **recalls,
        **ndcgs,
        "text": query["text"],
        "title": query["title"],
        "speaker": query["speaker"],
        "speaking_style": query["speaking_style"],
        "environmental_sound": query["environmental_sound"],
        "relevance": query["relevance"],
        "instruction": query["instruction"],
        "reranked-100-docs": reranked_docs_ids[:100],
    }
    json.dump(result_entry, file, ensure_ascii=False)
    file.write("\n")
