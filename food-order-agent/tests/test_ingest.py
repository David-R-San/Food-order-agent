"""Schema and price parsing against the real menu CSV."""

from pathlib import Path

import pandas as pd

from src.ingest import (
    EXPECTED_COLUMNS,
    document_from_row,
    documents_from_frame,
    load_menu_frame,
    parse_price,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "menu_items.csv"


def test_real_csv_shape() -> None:
    frame = load_menu_frame(CSV_PATH)
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 5213


def test_parse_price_formats_seen_in_the_csv() -> None:
    assert parse_price("$4") == (4.0, 4.0)
    assert parse_price("7.50") == (7.5, 7.5)
    assert parse_price("$5 sm/$8 lg") == (5.0, 8.0)
    assert parse_price("$15/$22.50") == (15.0, 22.5)
    assert parse_price("16 / 62") == (16.0, 62.0)
    assert parse_price("7.25/8.75") == (7.25, 8.75)
    # Counts next to a dollar amount are not prices.
    assert parse_price("2-4 people $9.50 / 5-7 people $12") == (9.5, 12.0)
    assert parse_price("14 oz $6.25 / 22 oz $7.75") == (6.25, 7.75)
    assert parse_price("$6.55+") == (6.55, 6.55)
    assert parse_price("35 prix-fixe") == (35.0, 35.0)
    assert parse_price("MP") == (None, None)
    assert parse_price("Market Price") == (None, None)
    assert parse_price("(market price)") == (None, None)
    assert parse_price("Ask us") == (None, None)
    assert parse_price("") == (None, None)


def test_vintage_year_is_not_the_price_when_another_amount_exists() -> None:
    assert parse_price("2024 16 / 62") == (16.0, 62.0)


def test_document_uses_type_as_cuisine_and_keeps_raw_price() -> None:
    document = document_from_row(
        {
            "restaurant": "Rasta Belly Jamaican Restaurant",
            "type": "Caribbean",
            "city": "Enterprise",
            "state": "AL",
            "opened": "2026-08-22",
            "section": "Rasta Belly Starters",
            "item": "Caribbean Chicken Soup",
            "description": "Slow simmered chicken.",
            "price": "$5 sm/$8 lg",
            "source_url": "https://rastabelly.com/",
        }
    )
    assert "Nome: Caribbean Chicken Soup" in document.text
    assert "Culinária: Caribbean" in document.text
    assert "Descrição: Slow simmered chicken." in document.text
    assert document.metadata["cuisine"] == "Caribbean"
    assert document.metadata["price_raw"] == "$5 sm/$8 lg"
    assert document.metadata["price_min"] == 5.0
    assert document.metadata["has_price"] == 1
    assert document.item_id == document.metadata["item_id"]


def test_empty_description_is_omitted_and_missing_price_is_flagged() -> None:
    document = document_from_row(
        {
            "restaurant": "Azul Restaurant",
            "type": "Latin American",
            "city": "Hackensack",
            "state": "NJ",
            "opened": "2026-01-01",
            "section": "Wine",
            "item": "White",
            "description": "",
            "price": "MP",
            "source_url": "https://example.com",
        }
    )
    assert "Descrição:" not in document.text
    assert document.metadata["has_price"] == 0
    assert document.metadata["price_min"] == -1.0


def test_full_csv_item_ids_are_unique_and_cover_numeric_prices() -> None:
    frame = load_menu_frame(CSV_PATH)
    documents = documents_from_frame(frame)
    assert len(documents) == 5213
    assert len({document.item_id for document in documents}) == 5213
    missing_price = [
        document for document in documents if document.metadata["has_price"] == 0
    ]
    # 83 blank prices plus 6 non-numeric cells (MP / Market Price / Ask us).
    assert len(missing_price) == 89


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "menu.csv"
    pd.DataFrame({"name": ["Burger"], "cost": ["9"]}).to_csv(path, index=False)
    try:
        load_menu_frame(path)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
