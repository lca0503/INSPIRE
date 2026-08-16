"""Shared file-loading utilities for text extractors."""

import os

SPLITS = ("DailyTalk", "Expresso", "VCTK", "Synthetic")


def load_text_files(directory):
    """Load text files into a dictionary keyed by filename stem."""
    texts = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as file:
            texts[filename.split(".")[0]] = file.read()
    return texts
