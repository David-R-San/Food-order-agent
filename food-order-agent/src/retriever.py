"""Semantic search over menu documents, with cuisine and price filters.

Chroma stores vectors we computed ourselves with a multilingual MiniLM.
Query-time embeddings use the same model, so a Portuguese question can rank
English menu text. Similarity is cosine similarity in [0, 1], derived from
Chroma's cosine distance (1 - distance).
"""

from __future__ import annotations

import logging
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME, DEFAULT_TOP_K, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Portuguese (and a few loose English) names -> a fragment of the English
# `type` label that actually exists in the CSV. The dataset has no PT column.
CUISINE_ALIASES: dict[str, str] = {
    "italiana": "italian",
    "italiano": "italian",
    "mexicana": "mexican",
    "mexicano": "mexican",
    "tailandesa": "thai",
    "japonesa": "sushi",
    "japones": "sushi",
    "coreana": "korean",
    "churrasco": "bbq",
    "mediterranea": "mediterranean",
    "mediterraneo": "mediterranean",
    "espanhola": "spanish",
    "espanhol": "spanish",
    "francesa": "french",
    "frances": "french",
    "peruana": "peruvian",
    "peruano": "peruvian",
    "vietnamita": "vietnamese",
    "etiope": "ethiopian",
    "etiopia": "ethiopian",
    "colombiana": "colombian",
    "colombiano": "colombian",
    "indiana": "indian",
    "indiano": "indian",
    "africana": "african",
    "africano": "african",
    "hamburguer": "burgers",
    "hamburger": "burgers",
    "cafe": "coffee",
    "cafeteria": "coffee",
    "padaria": "bakery",
    "sanduiche": "sandwiches",
    "sanduiches": "sandwiches",
    "saudavel": "healthy",
    "acai": "acai",
    "caribenha": "caribbean",
    "caribenho": "caribbean",
    "texmex": "tex-mex",
    "latina": "latin",
    "latino": "latin",
    "asas": "wings",
    "europeia": "european",
    "europeu": "european",
    "sulista": "southern",
    "cafe da manha": "breakfast",
    "cha": "chai",
    "bubble tea": "bubble tea",
    "bubbletea": "bubble tea",
}


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local multilingual embeddings. The model name comes from the environment."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        # Large default thread pools can oversubscribe CPUs on Windows laptops.
        torch.set_num_threads(int(os.getenv("EMBEDDING_NUM_THREADS", "4")))
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            # Larger batches reduce Python overhead for this ~5k-row menu.
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)
        return vector[0].tolist()


@dataclass(frozen=True)
class RetrievedItem:
    item_id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool]

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", ""))

    @property
    def restaurant(self) -> str:
        return str(self.metadata.get("restaurant", ""))

    @property
    def cuisine(self) -> str:
        return str(self.metadata.get("cuisine", ""))

    @property
    def price_raw(self) -> str:
        return str(self.metadata.get("price_raw", ""))


@dataclass
class RetrievalResponse:
    hits: list[RetrievedItem] = field(default_factory=list)
    # unknown_cuisine | no_matches | below_min_score
    notice: str | None = None
    notice_detail: str | None = None


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_label(text: str) -> str:
    return " ".join(strip_accents(text).casefold().split())


def resolve_cuisines(query: str, known: list[str]) -> list[str]:
    """Map a user cuisine string onto labels that exist in the index.

    Fragments shorter than 3 characters must match the whole label. That keeps
    a stray "a" from selecting every cuisine that contains the letter.
    """
    label = normalize_label(query)
    if not label:
        return []
    fragment = CUISINE_ALIASES.get(label, label)
    matches: list[str] = []
    for cuisine in known:
        known_label = normalize_label(cuisine)
        if fragment == known_label:
            matches.append(cuisine)
            continue
        if len(fragment) < 3:
            continue
        if fragment in known_label or known_label in fragment:
            matches.append(cuisine)
    return matches


def distance_to_similarity(distance: float) -> float:
    """Chroma cosine distance is 1 - cosine similarity."""
    similarity = 1.0 - float(distance)
    return max(0.0, min(1.0, similarity))


class MenuRetriever:
    def __init__(
        self,
        persist_dir: Path | str = CHROMA_DIR,
        embedder: Embedder | None = None,
        collection_name: str = COLLECTION_NAME,
        client=None,
        collection=None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.embedder = embedder or SentenceTransformerEmbedder(EMBEDDING_MODEL)
        # Reuse the client that just wrote the index. A second PersistentClient
        # on the same path fails on Windows while the first one still holds the lock.
        if client is not None and collection is not None:
            self.client = client
            self.collection = collection
        else:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_collection(collection_name)
        self._cuisines: list[str] | None = None

    def count(self) -> int:
        return int(self.collection.count())

    def known_cuisines(self) -> list[str]:
        if self._cuisines is None:
            payload = self.collection.get(include=["metadatas"])
            found = {
                str(metadata["cuisine"])
                for metadata in payload.get("metadatas") or []
                if metadata and metadata.get("cuisine")
            }
            self._cuisines = sorted(found)
        return self._cuisines

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        max_price: float | None = None,
        cuisine: str | None = None,
        min_score: float | None = None,
    ) -> RetrievalResponse:
        """Return the top-k items and their cosine similarity.

        `max_price` compares against `price_min` (the cheapest printed amount)
        and skips items with no numeric price. An unknown cuisine does not
        fall back to an unfiltered search.
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        where_clauses: list[dict] = []
        if cuisine:
            matched = resolve_cuisines(cuisine, self.known_cuisines())
            if not matched:
                return RetrievalResponse(
                    notice="unknown_cuisine",
                    notice_detail=cuisine,
                )
            where_clauses.append({"cuisine": {"$in": matched}})

        if max_price is not None:
            where_clauses.append({"has_price": {"$eq": 1}})
            where_clauses.append({"price_min": {"$lte": float(max_price)}})

        where = _combine_where(where_clauses)
        total = self.count()
        if total == 0:
            return RetrievalResponse(notice="no_matches")

        # Chroma errors when n_results is larger than the collection.
        n_results = min(k, total)
        query_vector = self.embedder.embed_query(query)
        try:
            raw = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Chroma query failed: %s", exc)
            message = str(exc).lower()
            if "0 elements" in message or "not enough" in message or "k:" in message:
                return RetrievalResponse(notice="no_matches")
            raise

        hits = _hits_from_query(raw)
        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= min_score]
            if not hits:
                return RetrievalResponse(notice="below_min_score", notice_detail=query)
        if not hits:
            return RetrievalResponse(notice="no_matches")
        return RetrievalResponse(hits=hits)


def _combine_where(clauses: list[dict]) -> dict | None:
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _hits_from_query(raw: dict) -> list[RetrievedItem]:
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    hits: list[RetrievedItem] = []
    for item_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        meta = dict(metadata or {})
        resolved_id = str(meta.get("item_id") or item_id)
        hits.append(
            RetrievedItem(
                item_id=resolved_id,
                text=document or "",
                score=distance_to_similarity(distance),
                metadata=meta,
            )
        )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits


def index_documents(
    documents,
    *,
    persist_dir: Path | str,
    embedder: Embedder,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 1000,
) -> MenuRetriever:
    """Replace the collection with `documents` and return a retriever over it."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    existing = {collection.name for collection in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedder.model_name,
        },
    )
    texts = [document.text for document in documents]
    vectors = embedder.embed_documents(texts) if texts else []
    if len(vectors) != len(documents):
        raise RuntimeError("Embedder returned a different number of vectors than documents")

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        batch_vectors = vectors[start : start + batch_size]
        collection.add(
            ids=[document.item_id for document in batch],
            documents=[document.text for document in batch],
            metadatas=[document.metadata for document in batch],
            embeddings=batch_vectors,
        )

    return MenuRetriever(
        persist_dir=persist_path,
        embedder=embedder,
        collection_name=collection_name,
        client=client,
        collection=collection,
    )
