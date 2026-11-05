"""Clean the public menu CSV and build the Chroma index.

Dataset (read before this schema was written):
https://www.248.ai/blog/every-item-on-the-menu-at-50-new-restaurants

    data/raw/menu_items.csv
    5,213 rows, 10 columns, about 1.0 MB

    restaurant, type, city, state, opened, section, item, description, price, source_url

There is no separate cuisine or numeric-price column. `type` is the cuisine
label (40 English values such as "Thai" or "Mexican cafe"). `price` is free
text in US dollars ("$12", "16 / 62", "Market Price"). 2,355 descriptions and
83 prices are empty. Item names are never empty. The key
restaurant+section+item+price is unique.

`price_min` is the cheapest amount printed in the cell (glass vs bottle,
small vs large). A max-price filter uses that cheapest amount, so a ranged
item stays available when one of its options fits the budget. Cells with no
number ("MP", "Ask us") get has_price=0 and drop out only when a max price
is requested.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass

import pandas as pd

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RAW_CSV
from src.retriever import Embedder, MenuRetriever, index_documents

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "restaurant",
    "type",
    "city",
    "state",
    "opened",
    "section",
    "item",
    "description",
    "price",
    "source_url",
]

# Dollar amounts win over nearby counts ("2-4 people $9.50", "14 oz $6.25").
_DOLLAR_AMOUNT = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_DECIMAL_AMOUNT = re.compile(r"\d+\.\d+")
_INTEGER_AMOUNT = re.compile(r"\d+")


@dataclass(frozen=True)
class MenuDocument:
    """One menu row ready for the vector index."""

    item_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


def load_menu_frame(path=RAW_CSV) -> pd.DataFrame:
    """Load the CSV as strings. Empty cells stay empty; nothing is imputed."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [column.strip() for column in frame.columns]
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        found = ", ".join(frame.columns)
        raise ValueError(f"CSV is missing columns {missing}. Found: {found}")
    return frame


def parse_price(raw: object) -> tuple[float | None, float | None]:
    """Return (lowest amount, highest amount) printed in a price cell.

    Years (1900-2100) are dropped when another amount remains, because a
    vintage in the same cell is not a price. If every number looks like a
    year, the original numbers are kept.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None, None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None, None

    if _DOLLAR_AMOUNT.search(text):
        amounts = [float(match) for match in _DOLLAR_AMOUNT.findall(text)]
    elif _DECIMAL_AMOUNT.search(text):
        amounts = [float(match) for match in _DECIMAL_AMOUNT.findall(text)]
    else:
        amounts = [float(match) for match in _INTEGER_AMOUNT.findall(text)]

    amounts = _drop_vintages(amounts)
    if not amounts:
        return None, None
    return min(amounts), max(amounts)


def _drop_vintages(amounts: list[float]) -> list[float]:
    kept = [
        amount
        for amount in amounts
        if not (amount.is_integer() and 1900 <= amount <= 2100)
    ]
    return kept or amounts


def make_item_id(restaurant: str, section: str, item: str, price: str) -> str:
    """Stable id. Those four fields are unique in the source CSV."""
    raw = "\n".join([restaurant, section, item, price])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def document_from_row(row: dict[str, str]) -> MenuDocument:
    """Turn one CSV row into embedding text plus filter metadata.

    The embedded text uses the five fields the product searches on. City,
    section, and the source URL stay in metadata for display and filters.
    """
    restaurant = row["restaurant"].strip()
    cuisine = row["type"].strip()
    item = row["item"].strip()
    description = row["description"].strip()
    price_raw = row["price"].strip()
    section = row["section"].strip()
    price_min, price_max = parse_price(price_raw)

    lines = [f"Nome: {item}"]
    if description:
        lines.append(f"Descrição: {description}")
    lines.extend(
        [
            f"Restaurante: {restaurant}",
            f"Culinária: {cuisine}",
            f"Preço: {price_raw or 'não informado'}",
        ]
    )

    metadata: dict[str, str | int | float | bool] = {
        "item_id": make_item_id(restaurant, section, item, price_raw),
        "name": item,
        "description": description,
        "restaurant": restaurant,
        "cuisine": cuisine,
        "section": section,
        "city": row["city"].strip(),
        "state": row["state"].strip(),
        "opened": row["opened"].strip(),
        "price_raw": price_raw,
        # Chroma metadata cannot store None. -1 means "no numeric price".
        "price_min": -1.0 if price_min is None else float(price_min),
        "price_max": -1.0 if price_max is None else float(price_max),
        "has_price": 0 if price_min is None else 1,
        "source_url": row["source_url"].strip(),
    }
    return MenuDocument(
        item_id=str(metadata["item_id"]),
        text="\n".join(lines),
        metadata=metadata,
    )


def documents_from_frame(frame: pd.DataFrame) -> list[MenuDocument]:
    documents: list[MenuDocument] = []
    seen: set[str] = set()
    for record in frame.to_dict(orient="records"):
        document = document_from_row(record)
        if document.item_id in seen:
            raise ValueError(f"Duplicate item_id {document.item_id}")
        seen.add(document.item_id)
        documents.append(document)
    return documents


def build_index(
    frame: pd.DataFrame | None = None,
    *,
    persist_dir=CHROMA_DIR,
    embedder: Embedder | None = None,
    collection_name: str = COLLECTION_NAME,
) -> MenuRetriever:
    """Embed every menu item and persist a cosine Chroma collection."""
    if frame is None:
        frame = load_menu_frame()
    if embedder is None:
        from src.retriever import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL)

    documents = documents_from_frame(frame)
    logger.info("Indexing %s menu documents", len(documents))
    return index_documents(
        documents,
        persist_dir=persist_dir,
        embedder=embedder,
        collection_name=collection_name,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = load_menu_frame()
    print(f"rows={len(frame)}")
    print(f"columns={list(frame.columns)}")
    print(frame.head(5).to_string(index=False))
    retriever = build_index(frame)
    print(f"indexed={retriever.count()} collection={COLLECTION_NAME} dir={CHROMA_DIR}")


if __name__ == "__main__":
    main()
