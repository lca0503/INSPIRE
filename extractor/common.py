"""Shared in-memory audio preparation."""

import librosa


def resample_audio(item, target_sampling_rate):
    """Return an item's audio array resampled to the model sampling rate."""
    audio = item["audio"]
    return librosa.resample(
        audio["array"],
        orig_sr=audio["sampling_rate"],
        target_sr=target_sampling_rate,
    )
