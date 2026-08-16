import argparse
import json
import os
import random
from statistics import fmean

from calculate_score import SPLITS, calculate_ndcg_at_k, calculate_recall_at_k
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate recall@k and NDCG@k metrics using random baseline (averaged over multiple runs)"
    )
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of random runs to average over",
    )
    return parser.parse_args()


def random_retrieval(doc_ids, excluded_ids, seed=None):
    """Random retrieval."""
    filtered_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in excluded_ids]
    shuffled_ids = filtered_doc_ids.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled_ids)

    return shuffled_ids


def write_score_entry(file, query_id, query, doc_ids, k_values, num_runs, seed):
    """Calculate and write averaged random-baseline metrics for one query."""
    positive_docs = set(query["positive_documents"])
    excluded_ids = set(query["excluded_ids"])
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}

    for run in range(num_runs):
        retrieved_ids = random_retrieval(doc_ids, excluded_ids, seed=seed + run)
        for k in k_values:
            recalls[k].append(
                calculate_recall_at_k(retrieved_ids, positive_docs, k)
            )
            ndcgs[k].append(calculate_ndcg_at_k(retrieved_ids, positive_docs, k))

    result_entry = {
        "id": query_id,
        **{f"recall@{k}": float(fmean(recalls[k])) for k in k_values},
        **{f"ndcg@{k}": float(fmean(ndcgs[k])) for k in k_values},
        "text": query["text"],
        "title": query["title"],
        "speaker": query["speaker"],
        "speaking_style": query["speaking_style"],
        "environmental_sound": query["environmental_sound"],
        "relevance": query["relevance"],
        "instruction": query["instruction"],
    }
    json.dump(result_entry, file, ensure_ascii=False)
    file.write("\n")


def main(args):
    """Main function."""
    print(f"Loading dataset: {args.dataset_name}")
    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    k_values = sorted(args.k_values)

    for split in SPLITS:
        print(f"\nProcessing {split} split...")
        print(f"Find {len(queries[split])} queries in {split} split")
        print(f"Find {len(documents[split])} documents in {split} split")

        doc_ids = [document["audio_id"] for document in documents[split]]
        print(f"Found {len(doc_ids)} document IDs in {split} split")

        output_file = os.path.join(args.output_dir, split, "score.jsonl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as file:
            for idx, query in enumerate(
                tqdm(queries[split], desc="Processing queries")
            ):
                write_score_entry(
                    file,
                    f"{idx:05d}",
                    query,
                    doc_ids,
                    k_values,
                    args.num_runs,
                    args.seed + idx * args.num_runs,
                )

        print(f"Saved scores to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
