import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
from pyserini import analysis
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100]
    )

    return parser.parse_args()


def calculate_recall_at_k(retrieved_ids, positive_docs, k):
    """Calculate recall@k."""
    if len(positive_docs) == 0:
        return 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k_retrieved & positive_docs)
    recall = relevant_retrieved / len(positive_docs)
    return recall


def calculate_ndcg_at_k(retrieved_ids, positive_docs, k):
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain)."""
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


def prepare_documents(documents):
    """Prepare documents for BM25 retrieval."""
    doc_texts = []
    doc_ids = []
    for doc in documents:
        audio_id = doc["audio_id"]
        doc_transcription = doc["text"]
        doc_speaker = doc["speaker"]
        doc_speaking_style = doc["speaking_style"]
        doc_environmental_sound = doc["environmental_sound"]
        doc_caption = f"Speaker: {doc_speaker}, Speaking Style: {doc_speaking_style}, Environmental Sound: {doc_environmental_sound}"
        doc_text = f"Transcription: {doc_transcription} Caption: {doc_caption}"
        doc_texts.append(doc_text)
        doc_ids.append(audio_id)
    return doc_texts, doc_ids


def build_bm25_index(doc_texts, analyzer, k1, b):
    """Build BM25 index from document texts."""
    print("Building BM25 index...")
    corpus = [analyzer.analyze(x) for x in doc_texts]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=k1, b=b)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(
        bm25_corpus,
        num_docs=len(corpus),
        num_terms=len(dictionary),
        normalize_queries=False,
        normalize_documents=False,
    )
    return model, dictionary, bm25_index


def compute_query_scores(
    query_text, analyzer, model, dictionary, bm25_index, doc_ids, excluded_ids
):
    """Compute BM25 scores for a query and return sorted document IDs."""
    analyzed_query = analyzer.analyze(query_text)
    bm25_query = model[dictionary.doc2bow(analyzed_query)]
    similarities = bm25_index[bm25_query].tolist()

    all_scores = {}
    for did, s in zip(doc_ids, similarities):
        all_scores[did] = float(s)

    for did in set(excluded_ids):
        if did != "N/A" and did in all_scores:
            all_scores.pop(did)

    cur_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    retrieved_docs_ids = [doc_id for doc_id, _ in cur_scores]
    return retrieved_docs_ids


def calculate_metrics(retrieved_docs_ids, positive_docs, k_values):
    """Calculate recall@k and NDCG@k metrics."""
    query_recalls = {}
    query_ndcgs = {}
    for k in k_values:
        recall = calculate_recall_at_k(retrieved_docs_ids, positive_docs, k)
        query_recalls[f"recall@{k}"] = float(recall)

        ndcg = calculate_ndcg_at_k(retrieved_docs_ids, positive_docs, k)
        query_ndcgs[f"ndcg@{k}"] = float(ndcg)
    return query_recalls, query_ndcgs


def create_result_entry(
    query_id, query, retrieved_docs_ids, query_recalls, query_ndcgs
):
    """Create result entry for BM25 retrieval."""
    result_entry = {
        "id": query_id,
        **query_recalls,
        **query_ndcgs,
    }

    result_entry["text"] = query["text"]
    result_entry["title"] = query["title"]
    result_entry["speaker"] = query["speaker"]
    result_entry["speaking_style"] = query["speaking_style"]
    result_entry["environmental_sound"] = query["environmental_sound"]
    result_entry["relevance"] = query["relevance"]
    result_entry["instruction"] = query["instruction"]
    result_entry["top-100-retrieved-docs"] = retrieved_docs_ids[:100]

    return result_entry


def process_split(split, queries, documents, k1, b, analyzer, k_values, output_dir):
    """Process a single split: load data, build index, and process queries."""
    print(f"\nProcessing {split} split...")
    print(f"Find {len(queries[split])} queries in {split} split")
    print(f"Find {len(documents[split])} documents in {split} split")

    doc_texts, doc_ids = prepare_documents(documents[split])

    model, dictionary, bm25_index = build_bm25_index(doc_texts, analyzer, k1, b)

    output_file = os.path.join(output_dir, split, "score.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        bar = tqdm(queries[split], desc="BM25 retrieval")
        for idx, query in enumerate(bar):
            query_id = f"{idx:05d}"
            query_instruction = query["instruction"]
            query_transcription = query["text"]
            query_speaker = query["speaker"]
            query_speaking_style = query["speaking_style"]
            query_environmental_sound = query["environmental_sound"]
            query_caption = f"Speaker: {query_speaker}, Speaking Style: {query_speaking_style}, Environmental Sound: {query_environmental_sound}"

            query_text = f"Instruct: {query_instruction}\nQuery: Transcription: {query_transcription} Caption: {query_caption}"

            retrieved_docs_ids = compute_query_scores(
                query_text,
                analyzer,
                model,
                dictionary,
                bm25_index,
                doc_ids,
                query["excluded_ids"],
            )

            positive_docs_audio_ids = set(query["positive_documents"])

            query_recalls, query_ndcgs = calculate_metrics(
                retrieved_docs_ids, positive_docs_audio_ids, k_values
            )

            result_entry = create_result_entry(
                query_id, query, retrieved_docs_ids, query_recalls, query_ndcgs
            )

            json.dump(result_entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved scores to {output_file}")


def main(args):
    """Main function."""
    print(f"Loading dataset: {args.dataset_name}")
    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    k_values = sorted(args.k_values)
    splits = ["DailyTalk", "Expresso", "VCTK", "Synthetic"]

    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())

    for split in splits:
        process_split(
            split,
            queries,
            documents,
            args.k1,
            args.b,
            analyzer,
            k_values,
            args.output_dir,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
