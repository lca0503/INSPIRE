import argparse
import os

from common import SPLITS, build_caption_question, resample_audio, write_vllm_outputs
from datasets import load_dataset
from tqdm import tqdm
from transformers import set_seed
from vllm import LLM, SamplingParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lca0503/INSPIRE")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--use_detailed_caption", action="store_true")
    parser.add_argument("--queries_only", action="store_true")

    return parser.parse_args()


def prepare_inputs(item, sampling_rate, audio_count=1, use_detailed_caption=False):
    """Prepare model input prompt and audio data for a given item."""
    resampled_audio = resample_audio(item, sampling_rate)

    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )

    audio_in_prompt = "".join(
        ["<|audio_bos|><|AUDIO|><|audio_eos|>\n" for _ in range(audio_count)]
    )

    question = build_caption_question(item, use_detailed_caption)

    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    audios = [resampled_audio]

    return prompt, audios


def extract_captions(
    llm,
    items,
    items_dir,
    sampling_rate,
    sampling_params,
    audio_count=1,
    use_detailed_caption=False,
):
    """Extract captions for INSPIRE."""
    os.makedirs(items_dir, exist_ok=True)

    batch_inputs = []
    for idx, item in enumerate(tqdm(items, desc="Preparing inputs")):
        prompt, audios = prepare_inputs(
            item, sampling_rate, audio_count, use_detailed_caption
        )
        batch_inputs.append({"prompt": prompt, "multi_modal_data": {"audio": audios}})

    print(f"Generating captions for {len(batch_inputs)} items...")
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    write_vllm_outputs(outputs, items_dir)


def main(args):
    """Extract captions for INSPIRE."""
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    audio_count = 1

    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"audio": audio_count},
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    for split in SPLITS:
        print(f"\nProcessing {split} split...")

        queries_dir = os.path.join(args.output_dir, "queries", split)
        extract_captions(
            llm,
            queries[split],
            queries_dir,
            args.sampling_rate,
            sampling_params,
            audio_count,
            args.use_detailed_caption,
        )

        if not args.queries_only:
            documents_dir = os.path.join(args.output_dir, "documents", split)
            extract_captions(
                llm,
                documents[split],
                documents_dir,
                args.sampling_rate,
                sampling_params,
                audio_count,
                args.use_detailed_caption,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
