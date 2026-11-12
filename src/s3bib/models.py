"""Embedding models users can choose from. Any sentence-transformers model name also works."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

# key: (model id, kind, max tokens, description)
EMBEDDING_MODELS: Dict[str, Tuple[str, str, int, str]] = {
    "mpnet": (DEFAULT_MODEL, "text", 384,
              "Default. The model used in the published S3 paper; strong general-purpose English model."),
    "specter": ("sentence-transformers/allenai-specter", "scientific", 512,
                "Trained on scientific papers (titles + abstracts linked by citations)."),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "text", 256,
               "Small and about 5x faster than mpnet; slightly less accurate. Good for quick looks."),
    "multilingual": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "multilingual", 128,
                     "50+ languages, for corpora that mix languages. Reads only the first 128 tokens."),
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", "text", 32768,
              "Newer, larger model with long context (no abstract truncation); slower on CPU."),
    "clip": ("sentence-transformers/clip-ViT-B-32", "multimodal", 77,
             "Multimodal (text and images in one space). For titles + abstracts it reads only "
             "77 tokens, so use it only when you also embed images."),
}

# S3's threshold range 0.75-0.85 was calibrated for mpnet. Other models spread cosine
# similarities differently, so they get a wider scan unless the user sets thresholds.
MPNET_THRESHOLDS = tuple(round(float(t), 2) for t in np.arange(0.75, 0.86, 0.01))
WIDE_THRESHOLDS = tuple(round(float(t), 2) for t in np.arange(0.40, 0.96, 0.01))


def resolve_model(name_or_key: str | None) -> str:
    """'mpnet' -> 'sentence-transformers/all-mpnet-base-v2'; full model names pass through."""
    if not name_or_key:
        return DEFAULT_MODEL
    key = name_or_key.strip().lower()
    return EMBEDDING_MODELS[key][0] if key in EMBEDDING_MODELS else name_or_key.strip()


def default_thresholds(model_name: str) -> tuple:
    return MPNET_THRESHOLDS if resolve_model(model_name) == DEFAULT_MODEL else WIDE_THRESHOLDS


def model_table() -> List[Dict[str, object]]:
    return [{"key": k, "model": m, "kind": kind, "max_tokens": mt, "description": d,
             "default": m == DEFAULT_MODEL}
            for k, (m, kind, mt, d) in EMBEDDING_MODELS.items()]
