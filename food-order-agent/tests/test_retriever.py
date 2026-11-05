"""Retriever filters and scores, with a deterministic embedder (no model download)."""

from __future__ import annotations

import math
import re

from src.ingest import MenuDocument, build_index
from src.retriever import resolve_cuisines

_TOKEN = re.compile(r"[a-z0-9]+")


class OverlapEmbedder:
    """Bag-of-words embedder so tests can rank documents without MiniLM."""

    model_name = "test-overlap"

    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            for token in _tokenize(text):
                if token not in self._index:
                    self._index[token] = len(self._index)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        # Bias dimension keeps an empty query from becoming the zero vector.
        vector = [0.0] * (len(self._index) + 1)
        vector[-1] = 1.0
        for token in _tokenize(text):
            position = self._index.get(token)
            if position is not None:
                vector[position] = 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _row(
    *,
    restaurant: str,
    cuisine: str,
    item: str,
    description: str,
    price: str,
    section: str = "Mains",
) -> dict[str, str]:
    return {
        "restaurant": restaurant,
        "type": cuisine,
        "city": "Enterprise",
        "state": "AL",
        "opened": "2026-08-22",
        "section": section,
        "item": item,
        "description": description,
        "price": price,
        "source_url": "https://example.com",
    }


def _index(tmp_path, rows: list[dict[str, str]]):
    import pandas as pd

    from src.ingest import documents_from_frame

    frame = pd.DataFrame(rows)
    embedder = OverlapEmbedder()
    # Fit the vocabulary before queries that the documents never contained.
    documents = documents_from_frame(frame)
    return build_index(
        frame,
        persist_dir=tmp_path / "chroma",
        embedder=embedder,
    ), documents


def test_resolve_cuisines_portuguese_alias_and_unknown() -> None:
    known = ["Thai", "Mexican cafe", "Italian", "Korean BBQ", "Coffee"]
    assert resolve_cuisines("tailandesa", known) == ["Thai"]
    assert resolve_cuisines("mexicana", known) == ["Mexican cafe"]
    assert resolve_cuisines("italiana", known) == ["Italian"]
    assert resolve_cuisines("churrasco", known) == ["Korean BBQ"]
    assert resolve_cuisines("klingon", known) == []
    assert resolve_cuisines("a", known) == []


def test_search_ranks_overlap_and_returns_similarity(tmp_path) -> None:
    retriever, _documents = _index(
        tmp_path,
        [
            _row(
                restaurant="Rasta Belly",
                cuisine="Caribbean",
                item="Jerk Chicken",
                description="Smoky grilled jerk chicken with festival.",
                price="$12",
            ),
            _row(
                restaurant="Ambrosia",
                cuisine="Italian",
                item="Tiramisu",
                description="Espresso mascarpone dessert.",
                price="$9",
            ),
            _row(
                restaurant="Rasta Belly",
                cuisine="Caribbean",
                item="Jerk Wings",
                description="Smoky jerk wings.",
                price="$20",
            ),
        ],
    )
    response = retriever.search("jerk chicken", k=3)
    assert response.notice is None
    assert response.hits[0].name == "Jerk Chicken"
    assert response.hits[0].score >= response.hits[1].score
    assert 0.0 <= response.hits[0].score <= 1.0


def test_price_and_cuisine_filters(tmp_path) -> None:
    retriever, _documents = _index(
        tmp_path,
        [
            _row(
                restaurant="Rasta Belly",
                cuisine="Caribbean",
                item="Jerk Chicken",
                description="Smoky grilled jerk chicken.",
                price="$12",
            ),
            _row(
                restaurant="Rasta Belly",
                cuisine="Caribbean",
                item="Jerk Wings",
                description="Smoky jerk wings.",
                price="$20",
            ),
            _row(
                restaurant="Ambrosia",
                cuisine="Italian",
                item="Tiramisu",
                description="Espresso dessert.",
                price="$9",
            ),
            _row(
                restaurant="Azul",
                cuisine="Latin American",
                item="White",
                description="House white wine.",
                price="MP",
            ),
        ],
    )
    filtered = retriever.search("jerk", k=5, max_price=15, cuisine="caribenha")
    assert [hit.name for hit in filtered.hits] == ["Jerk Chicken"]
    assert filtered.hits[0].metadata["price_min"] == 12.0

    unknown = retriever.search("pizza", k=5, cuisine="marciana")
    assert unknown.hits == []
    assert unknown.notice == "unknown_cuisine"

    # No numeric price: visible without a cap, hidden when a cap is set.
    open_search = retriever.search("white wine", k=5, cuisine="latina")
    assert any(hit.name == "White" for hit in open_search.hits)
    capped = retriever.search("white wine", k=5, cuisine="latina", max_price=30)
    assert capped.hits == [] or all(hit.name != "White" for hit in capped.hits)
    assert all(hit.name != "White" for hit in capped.hits)


def test_min_score_hides_weak_matches(tmp_path) -> None:
    retriever, _documents = _index(
        tmp_path,
        [
            _row(
                restaurant="Ambrosia",
                cuisine="Italian",
                item="Tiramisu",
                description="Espresso mascarpone dessert.",
                price="$9",
            )
        ],
    )
    response = retriever.search("sushi de atum", k=5, min_score=0.99)
    assert response.hits == []
    assert response.notice == "below_min_score"


def test_documents_round_trip_metadata(tmp_path) -> None:
    retriever, documents = _index(
        tmp_path,
        [
            _row(
                restaurant="Ari",
                cuisine="Thai",
                item="Curry Noodles",
                description="Rice noodles in curry.",
                price="$22.99",
            )
        ],
    )
    assert retriever.count() == 1
    hit = retriever.search("curry noodles", k=1).hits[0]
    original: MenuDocument = documents[0]
    assert hit.item_id == original.item_id
    assert hit.cuisine == "Thai"
    assert hit.price_raw == "$22.99"
