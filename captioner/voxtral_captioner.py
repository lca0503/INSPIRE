import argparse
import os

from common import (
    SPLITS,
    audio_to_wav_bytes,
    build_caption_question,
    resample_audio,
    write_vllm_outputs,
)
from datasets import load_dataset
from mistral_common.protocol.instruct.chunk import AudioChunk, RawAudio, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
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

def prepare_audio_input(item, target_sampling_rate):
    """Prepare an in-memory Mistral audio input."""
    audio = resample_audio(item, target_sampling_rate)
    wav_bytes = audio_to_wav_bytes(audio, target_sampling_rate)
    return Audio.from_bytes(wav_bytes)

def prepare_inputs(
    item, tokenizer, model_id, sampling_rate, audio_count=1, use_detailed_caption=False
):
    """Prepare model input prompt and audio data for a given item."""
    audio = prepare_audio_input(item, sampling_rate)

    audio_chunks = [AudioChunk(input_audio=RawAudio.from_audio(audio))]

    question = build_caption_question(item, use_detailed_caption)

    text_chunk = TextChunk(text=question)

    messages = [UserMessage(content=[*audio_chunks, text_chunk])]

    req = ChatCompletionRequest(messages=messages, model=model_id)
    tokens = tokenizer.encode_chat_completion(req)
    prompt_token_ids, audios = tokens.tokens, tokens.audios

    audios_and_sr = [(au.audio_array, au.sampling_rate) for au in audios]

    return prompt_token_ids, audios_and_sr

def extract_captions(
    llm,
    tokenizer,
    model_id,
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
        prompt_token_ids, audios_and_sr = prepare_inputs(
            item, tokenizer, model_id, sampling_rate, audio_count, use_detailed_caption
        )
        batch_inputs.append(
            {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {"audio": audios_and_sr},
            }
        )

    print(f"Generating captions for {len(batch_inputs)} items...")
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    write_vllm_outputs(outputs, items_dir)

def main(args):
    """Extract captions for INSPIRE."""
    set_seed(args.seed)

    queries = load_dataset(args.dataset_name, "query")
    documents = load_dataset(args.dataset_name, "document")

    tokenizer = MistralTokenizer.from_hf_hub(args.model_name)

    audio_count = 1

    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"audio": audio_count},
        config_format="mistral",
        load_format="mistral",
        tokenizer_mode="mistral",
        enforce_eager=True,
        enable_chunked_prefill=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    for split in SPLITS:
        print(f"\nProcessing {split} split...")

        queries_dir = os.path.join(args.output_dir, "queries", split)
        extract_captions(
            llm,
            tokenizer,
            args.model_name,
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
                tokenizer,
                args.model_name,
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
