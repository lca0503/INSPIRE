"""Shared in-memory audio preparation."""

import io
import os

import librosa
import soundfile as sf

SPLITS = ("DailyTalk", "Expresso", "VCTK", "Synthetic")


def resample_audio(item, target_sampling_rate):
    """Return an item's audio array resampled to the model sampling rate."""
    audio = item["audio"]
    return librosa.resample(
        audio["array"],
        orig_sr=audio["sampling_rate"],
        target_sr=target_sampling_rate,
    )


def build_caption_question(item, use_detailed_caption):
    """Build the shared caption instruction for one dataset item."""
    if use_detailed_caption:
        return (
            "Generate a caption that describes the speech above, "
            "including its meaning, the speaker's identity or role, "
            "the speaking style, and any relevant environmental sounds."
        )
    question = "Generate a caption that describes the speech above."
    instruction = item.get("instruction")
    if instruction:
        question = (
            "Generate a caption that describes the speech above and can be used "
            f"for the following instruction: {instruction}"
        )
    return question


def audio_to_wav_bytes(audio, sampling_rate):
    """Encode an in-memory audio array as WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sampling_rate, format="WAV")
    return buffer.getvalue()


def write_vllm_outputs(outputs, output_dir):
    """Write vLLM caption outputs using five-digit filenames."""
    for index, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        output_path = os.path.join(output_dir, f"{index:05d}.txt")
        with open(output_path, "w") as file:
            file.write(generated_text)
