"""Paths and tunables loaded from the environment.

API keys are read from the environment only. Nothing secret is hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW_CSV = Path(os.getenv("MENU_CSV", ROOT / "data" / "raw" / "menu_items.csv"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", ROOT / "data" / "processed"))
CHROMA_DIR = PROCESSED_DIR / "chroma"
COLLECTION_NAME = "menu_items"

# Multilingual MiniLM: Portuguese queries over English menu text.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
