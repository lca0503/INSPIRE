import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

SPLITS = ("DailyTalk", "Expresso", "VCTK", "Synthetic")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate recall@k and NDCG@k metrics"
    )
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument(
        "--queries_representation_dir",
        required=True,
    )
    parser.add_argument(
        "--documents_representation_dir",
        required=True,
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export_file", type=str)
    parser.add_argument("--export_by_relevance_file", type=str)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100]
    )
    return parser.parse_args()


def load_representations(rep_dir):
    """Load representations."""
    reps_dict = {}
    rep_dir = Path(rep_dir)

    if not rep_dir.exists():
        raise FileNotFoundError(f"Representation directory not found: {rep_dir}")

    rep_files = list(rep_dir.glob("*.npy"))
    print(f"Loading {len(rep_files)} representations from {rep_dir}...")

    for rep_file in tqdm(rep_files, desc="Loading representations"):
        rep_id = rep_file.stem
        rep = np.load(rep_file)
        reps_dict[rep_id] = rep

    return reps_dict


def cosine_similarity(a, b):
    """Cosine similarity."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)


def compute_similarities(query_rep, docs_reps, rng):
    """Compute similarities with random tie-breaking."""
    similarities = [
        (doc_id, cosine_similarity(query_rep, doc_rep), rng.random())
        for doc_id, doc_rep in docs_reps.items()
    ]
    similarities.sort(key=lambda item: (-item[1], item[2]))
    return [(doc_id, similarity) for doc_id, similarity, _ in similarities]


def calculate_recall_at_k(retrieved_ids, positive_docs, k):
    """Recall@k."""
    if len(positive_docs) == 0:
        return 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k_retrieved & positive_docs)
    recall = relevant_retrieved / len(positive_docs)
    return recall


def calculate_ndcg_at_k(retrieved_ids, positive_docs, k):
    """NDCG@k."""
    if len(positive_docs) == 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in positive_docs:
            relevance = 1.0
            dcg += relevance / np.log2(i + 1)

    num_relevant = min(len(positive_docs), k)
    idcg = 0.0
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def load_scores(input_dir):
    """Load score files for all available splits."""
    scores = {}
    for split in SPLITS:
        score_file = Path(input_dir) / split / "score.jsonl"
        if score_file.exists():
            with open(score_file, encoding="utf-8") as file:
                scores[split] = [json.loads(line) for line in file if line.strip()]
    return scores


def mean_metric(entries, metric):
    values = [entry[metric] for entry in entries if metric in entry]
    return float(np.mean(values)) if values else 0.0


def export_scores(input_dir, output_file, k_values):
    """Export split and total score averages to CSV."""
    scores = load_scores(input_dir)
    all_entries = [entry for split in SPLITS for entry in scores.get(split, [])]
    rows = []
    for split in (*SPLITS, "Total"):
        entries = all_entries if split == "Total" else scores.get(split, [])
        if not entries:
            continue
        row = {"split": split}
        for k in k_values:
            row[f"recall@{k}"] = mean_metric(entries, f"recall@{k}")
            row[f"ndcg@{k}"] = mean_metric(entries, f"ndcg@{k}")
        row["num_queries"] = len(entries)
        rows.append(row)
    write_csv(output_file, rows)


def normalize_relevance(relevance):
    """Group specific style and sound relevance labels."""
    if not relevance:
        return relevance

    def replace(match):
        if match.group(1) == "same":
            return match.group(0)
        return f"specific {match.group(2)}"

    relevance = re.sub(
        r"([\w_]+)\s+(speaking style|environmental sound)", replace, relevance
    )
    return re.sub(r"\s+", " ", relevance.replace(" and and ", " and ")).strip()


def export_scores_by_relevance(input_dir, output_file, k=50):
    """Export recall grouped by split and relevance label."""
    grouped = defaultdict(list)
    metric = f"recall@{k}"
    for split, entries in load_scores(input_dir).items():
        for entry in entries:
            label = normalize_relevance(entry.get("relevance", "unknown"))
            grouped[(split, label)].append(entry.get(metric, 0.0))

    row = {}
    for split in SPLITS:
        labels = sorted(label for item_split, label in grouped if item_split == split)
        for label in labels:
            row[f"{split}_{label}_{metric}"] = float(np.mean(grouped[(split, label)]))
    write_csv(output_file, [row] if row else [])


def write_csv(output_file, rows):
    if not rows:
        print("No data found to export.")
        return
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {output_file}")


def write_score_entry(file, query_id, query, retrieved_docs_ids, k_values):
    """Calculate and write one query's score entry."""
    positive_docs = set(query["positive_documents"])
    recalls = {}
    ndcgs = {}

    for k in k_values:
        recalls[f"recall@{k}"] = float(
            calculate_recall_at_k(retrieved_docs_ids, positive_docs, k)
        )
        ndcgs[f"ndcg@{k}"] = float(
            calculate_ndcg_at_k(retrieved_docs_ids, positive_docs, k)
        )

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
        "top-100-retrieved-docs": retrieved_docs_ids[:100],
    }
    json.dump(result_entry, file, ensure_ascii=False)
    file.write("\n")


def main(args):
    """Main function."""
    print(f"Loading dataset: {args.dataset_name}")
    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    k_values = sorted(args.k_values)
    rng = random.Random(args.seed)

    for split in SPLITS:
        print(f"Find {len(queries[split])} queries in {split} split")
        print(f"Find {len(documents[split])} documents in {split} split")

        queries_dir = os.path.join(args.queries_representation_dir, split)
        documents_dir = os.path.join(args.documents_representation_dir, split)

        query_reps = load_representations(queries_dir)
        doc_reps = load_representations(documents_dir)

        print(f"Loaded {len(query_reps)} query representations in {split} split")
        print(f"Loaded {len(doc_reps)} document representations in {split} split")

        output_file = os.path.join(args.output_dir, split, "score.jsonl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as file:
            for idx, query in enumerate(
                tqdm(queries[split], desc="Processing queries")
            ):
                query_id = f"{idx:05d}"
                query_rep = query_reps[query_id]
                excluded_docs_audio_ids = set(query["excluded_ids"])
                filtered_docs_reps = {
                    doc_id: rep
                    for doc_id, rep in doc_reps.items()
                    if doc_id not in excluded_docs_audio_ids
                }

                similarities = compute_similarities(query_rep, filtered_docs_reps, rng)
                retrieved_docs_ids = [doc_id for doc_id, _ in similarities]

                write_score_entry(file, query_id, query, retrieved_docs_ids, k_values)

    if args.export_file:
        export_scores(args.output_dir, args.export_file, k_values)
    if args.export_by_relevance_file:
        export_scores_by_relevance(args.output_dir, args.export_by_relevance_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
