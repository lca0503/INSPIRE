import argparse
import os

import torch
from common import (
    SPLITS,
    load_top_retrieved_docs,
    resample_audio,
    write_rerank_entry,
)
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    set_seed,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rerank documents using Qwen3 Omni model"
    )
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100]
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def rerank(model, processor, query, documents, top_retrieved_docs):
    """Rerank documents."""
    tokenizer = getattr(processor, "tokenizer", processor)
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]

    target_sampling_rate = processor.feature_extractor.sampling_rate
    query_audio = resample_audio(query, target_sampling_rate)

    rerank_scores = []
    for doc_id in top_retrieved_docs:
        doc = documents[int(doc_id)]
        doc_audio = resample_audio(doc, target_sampling_rate)

        user_content = [
            {
                "type": "text",
                "text": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
            },
            {"type": "text", "text": "Instruct: " + query["instruction"]},
            {"type": "text", "text": "Query: "},
            {"type": "audio", "audio": query_audio},
            {"type": "text", "text": "Document: "},
            {"type": "audio", "audio": doc_audio},
        ]

        conversations = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True, return_dict=True)

        logits = output.logits[:, -1, :]
        yes_logit = logits[:, yes_id].squeeze(-1)
        no_logit = logits[:, no_id].squeeze(-1)
        score = torch.sigmoid(yes_logit - no_logit).item()
        rerank_scores.append((doc_id, score))

    rerank_scores.sort(key=lambda x: x[1], reverse=True)

    return rerank_scores


def main(args):
    """Main function."""
    set_seed(args.seed)

    print(f"Loading dataset: {args.dataset_name}")
    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    k_values = sorted(args.k_values)

    for split in SPLITS:
        print(f"Find {len(queries[split])} queries in {split} split")
        print(f"Find {len(documents[split])} documents in {split} split")

        top_retrieved_docs = load_top_retrieved_docs(
            os.path.join(args.input_dir, split, "score.jsonl")
        )

        output_file = os.path.join(args.output_dir, split, "score.jsonl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            for idx, query in enumerate(tqdm(queries[split], desc="Reranking queries")):
                top_retrieved_docs_ids = top_retrieved_docs[f"{idx:05d}"]

                rerank_scores = rerank(
                    model, processor, query, documents[split], top_retrieved_docs_ids
                )
                reranked_docs_ids = [doc_id for doc_id, _ in rerank_scores]

                write_rerank_entry(f, f"{idx:05d}", query, reranked_docs_ids, k_values)


if __name__ == "__main__":
    args = parse_args()
    main(args)
