# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""RAID adapter tests — synthetic rows only. No RAID text ever enters this repo; the
fixtures fabricate rows in RAID's verified column shape and the hub path is exercised
through an injected fetcher, never the network."""

from __future__ import annotations

import csv
import json
import random

import pytest

from gaige import corpus_raid
from gaige.corpus_raid import (
    Cell,
    EXPECTED_COLUMNS,
    RaidFetchError,
    RaidSchemaChanged,
    _map_row,
    _validate_columns,
    prepare_raid_slice,
    sample_csv,
)
from gaige.subgroups import auto_keys


def synth_raid_row(i: int, model: str, domain: str, attack: str = "none",
                   n_words: int = 80) -> dict:
    return {
        "id": f"row-{model}-{domain}-{attack}-{i}",
        "adv_source_id": "x", "source_id": "x",
        "model": model,
        "decoding": "" if model == "human" else "sampling",
        "repetition_penalty": "",
        "attack": attack,
        "domain": domain,
        "title": "t", "prompt": "",
        "generation": " ".join(f"synthetic{j}" for j in range(n_words)),
    }


def write_raid_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(EXPECTED_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_map_row_labels_and_meta_fills():
    h = _map_row(synth_raid_row(1, "human", "abstracts"))
    a = _map_row(synth_raid_row(2, "gpt4", "reddit", attack="homoglyph"))
    assert h["label"] == "human" and a["label"] == "ai"
    # Every meta key present on every row — subgroups.auto_keys requires universality.
    for m in (h, a):
        assert set(m["meta"]) == {"generator", "domain", "attack", "decoding"}
    assert h["meta"]["decoding"] == "none"  # empty upstream field → explicit "none"
    assert a["meta"]["attack"] == "homoglyph"


def test_map_row_refuses_empty_text():
    r = synth_raid_row(1, "gpt4", "reddit")
    r["generation"] = "   "
    assert _map_row(r) is None


def test_validate_columns_names_the_drift():
    with pytest.raises(RaidSchemaChanged, match="attack"):
        _validate_columns(["id", "model", "domain", "generation", "decoding"])


def test_csv_reservoir_is_seeded_deterministic(tmp_path):
    rows = [synth_raid_row(i, "human", "abstracts") for i in range(200)]
    rows += [synth_raid_row(i, "gpt4", "abstracts") for i in range(200)]
    p = tmp_path / "raid.csv"
    write_raid_csv(p, rows)
    cells = [Cell("human", "abstracts", "none"), Cell("gpt4", "abstracts", "none")]
    a = sample_csv(p, cells, 30, random.Random(7), min_words=50, max_words=500)
    b = sample_csv(p, cells, 30, random.Random(7), min_words=50, max_words=500)
    c = sample_csv(p, cells, 30, random.Random(8), min_words=50, max_words=500)
    ids = lambda pools: {k: [r["id"] for r in v] for k, v in pools.items()}  # noqa: E731
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_prepare_from_csv_end_to_end(tmp_path):
    rows = []
    for d in ("abstracts", "reddit"):
        rows += [synth_raid_row(i, "human", d) for i in range(120)]
        rows += [synth_raid_row(i, "gpt4", d) for i in range(120)]
    p = tmp_path / "raid.csv"
    write_raid_csv(p, rows)
    c = prepare_raid_slice(
        tmp_path / "corpora", generators=["gpt4"], domains=["abstracts", "reddit"],
        attacks=["none"], per_cell=60, seed=17, source="csv", raid_csv=p,
        progress=lambda s: None,
    )
    assert c.counts == {"human": 120, "ai": 120}
    assert c.path.name.startswith("raid-g1d2a1-n60-s17")
    # provenance lands in Corpus.meta → env.json verbatim
    for key in ("dataset", "dataset_revision", "selection", "filters",
                "retrieved_utc", "citation", "license", "note"):
        assert key in c.meta, key
    assert c.meta["dataset_revision"] == "local-csv"
    # universal meta keys → subgroup axes discovered
    items = [{"label": r["label"], "score": 0.0, "n_words": 80, "meta": r["meta"]}
             for r in c.items]
    assert set(auto_keys(items)) >= {"length_bucket", "generator", "domain",
                                     "attack", "decoding"}


def test_prepare_csv_without_path_refuses():
    with pytest.raises(RaidFetchError, match="11.8 GB"):
        prepare_raid_slice(
            __import__("pathlib").Path("/nonexistent-out"), generators=["gpt4"],
            domains=["abstracts"], attacks=["none"], source="csv", raid_csv=None,
        )


def test_hub_path_with_injected_fetcher(tmp_path):
    pool = {
        ("human", "abstracts"): [synth_raid_row(i, "human", "abstracts") for i in range(150)],
        ("gpt4", "abstracts"): [synth_raid_row(i, "gpt4", "abstracts") for i in range(150)],
    }

    def fake_fetch(url, timeout=30.0):
        if url == corpus_raid.REPO_API:
            return {"sha": "fakesha123"}
        # crude parse of the where clause + offset out of the filter url
        from urllib.parse import parse_qs, unquote, urlparse
        q = parse_qs(urlparse(url).query)
        where = unquote(q["where"][0])
        model = where.split("\"model\"='")[1].split("'")[0]
        domain = where.split("\"domain\"='")[1].split("'")[0]
        offset = int(q["offset"][0]); length = int(q["length"][0])
        rows = pool[(model, domain)]
        return {
            "features": [{"name": n} for n in EXPECTED_COLUMNS],
            "num_rows_total": len(rows),
            "rows": [{"row": r} for r in rows[offset:offset + length]],
        }

    c = prepare_raid_slice(
        tmp_path / "corpora", generators=["gpt4"], domains=["abstracts"],
        attacks=["none"], per_cell=60, seed=17, source="hub", fetch=fake_fetch,
        progress=lambda s: None,
    )
    assert c.counts == {"human": 60, "ai": 60}
    assert c.meta["dataset_revision"] == "fakesha123"
    # jsonl on disk round-trips through the standard loader (it IS the standard loader)
    first = json.loads(c.path.read_text(encoding="utf-8").splitlines()[0])
    assert set(first) == {"id", "text", "label", "meta"}


def test_word_filter_enforced(tmp_path):
    rows = [synth_raid_row(i, "human", "abstracts", n_words=10) for i in range(100)]
    rows += [synth_raid_row(100 + i, "human", "abstracts", n_words=80) for i in range(60)]
    p = tmp_path / "raid.csv"
    write_raid_csv(p, rows)
    pools = sample_csv(p, [Cell("human", "abstracts", "none")], 60, random.Random(1),
                       min_words=50, max_words=500)
    got = pools["human|abstracts|none"]
    assert len(got) == 60
    assert all(len(r["text"].split()) >= 50 for r in got)
