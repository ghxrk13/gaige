# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""RAID corpus adapter — prepare calibration slices from the RAID benchmark.

RAID (Dugan et al., ACL 2024, arXiv:2405.07940) is the largest shared benchmark for
machine-generated-text detection: 10M+ documents across 11 generators, 8 domains, 11
adversarial attacks, 4 decoding strategies. gaige never redistributes it: this module
FETCHES rows at preparation time (Hugging Face datasets-server pages, or a locally
downloaded RAID csv you provide) and writes a seeded slice under your gitignored
``corpora/`` directory. Every slice records full provenance — dataset revision sha,
selection parameters, retrieval time — because a calibration number without its corpus
identity is not a receipt.

Row mapping (column names verified against dataset revision 865cac74, 2026-07-25;
``_validate_columns`` refuses drift): ``model == "human"`` → label "human", anything else
→ label "ai"; text comes from ``generation``; ``meta = {generator, domain, attack,
decoding}`` on EVERY row — subgroup axis discovery requires universal presence
(subgroups.auto_keys), so human rows carry ``attack: "none"``, ``decoding: "none"``
rather than omitting keys.

The full train split is 11.8 GB — the hub path exists so a laptop can prepare an honest
slice without that download. At real RAID scale, download their csv and use
``--source csv`` (and expect batched scoring to matter; see the map — banked with its
design constraints until that day).
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import corpus

DATASET = "liamdugan/raid"
API_BASE = "https://datasets-server.huggingface.co"
REPO_API = f"https://huggingface.co/api/datasets/{DATASET}"

# Verified live against the dataset on 2026-07-25 (revision 865cac74). If RAID ever
# reshapes, _validate_columns names the drift instead of mis-mapping silently.
EXPECTED_COLUMNS = (
    "id",
    "adv_source_id",
    "source_id",
    "model",
    "decoding",
    "repetition_penalty",
    "attack",
    "domain",
    "title",
    "prompt",
    "generation",
)
CITATION = "Dugan et al., RAID, ACL 2024 (arXiv:2405.07940)"
LICENSE_NOTE = (
    "RAID is distributed by its authors on the Hugging Face Hub under their terms; "
    "gaige fetches rows at preparation time and redistributes none of it."
)

PAGE = 100  # datasets-server maximum rows per request
MAX_FETCH_TRIES = 4


class RaidSchemaChanged(RuntimeError):
    """The upstream RAID schema no longer matches what this adapter was verified on."""


class RaidFetchError(RuntimeError):
    """datasets-server could not be reached or answered unusably."""


@dataclass
class Cell:
    """One sampling cell: a (generator, domain, attack) combination."""

    model: str
    domain: str
    attack: str

    def key(self) -> str:
        return f"{self.model}|{self.domain}|{self.attack}"


def _validate_columns(names: list[str] | tuple[str, ...]) -> None:
    missing = [
        c for c in ("id", "model", "attack", "domain", "generation", "decoding") if c not in names
    ]
    if missing:
        raise RaidSchemaChanged(
            f"RAID columns missing {missing} (got {sorted(names)}). The upstream schema "
            "has changed since this adapter was verified (revision 865cac74); re-verify "
            "the mapping in corpus_raid.py before preparing any slice."
        )


def _map_row(row: dict) -> dict | None:
    """RAID row → gaige corpus row. Returns None for rows with no usable text."""
    text = row.get("generation") or ""
    if not text.strip():
        return None
    model = row.get("model") or "unknown"
    return {
        "id": str(row.get("id") or ""),
        "text": text,
        "label": "human" if model == "human" else "ai",
        "meta": {
            "generator": model,
            "domain": str(row.get("domain") or "unknown"),
            "attack": str(row.get("attack") or "none"),
            "decoding": str(row.get("decoding") or "none"),
        },
    }


def _word_ok(text: str, min_words: int, max_words: int) -> bool:
    n = len(text.split())
    return min_words <= n <= max_words


# --------------------------------------------------------------------------- hub path


def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    last = None
    for attempt in range(MAX_FETCH_TRIES):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                last = f"429 rate-limited (attempt {attempt + 1})"
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:  # noqa: PERF203 — retry loop is the point
            last = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RaidFetchError(f"{url} failed after {MAX_FETCH_TRIES} tries: {last}")


def _where(cell: Cell) -> str:
    # datasets-server /filter SQL-ish where clause; values are controlled vocabulary
    # (RAID's own category strings), quoted simply.
    return f"\"model\"='{cell.model}' AND \"domain\"='{cell.domain}' AND \"attack\"='{cell.attack}'"


def _filter_url(cell: Cell, offset: int, length: int) -> str:
    from urllib.parse import quote

    return (
        f"{API_BASE}/filter?dataset={quote(DATASET, safe='')}&config=raid"
        f"&split=train&where={quote(_where(cell), safe='')}"
        f"&offset={offset}&length={length}"
    )


def _revision_sha(fetch=_fetch_json) -> str:
    return str(fetch(REPO_API).get("sha") or "unknown")


def sample_cell_hub(
    cell: Cell,
    need: int,
    rng: random.Random,
    *,
    min_words: int,
    max_words: int,
    fetch=_fetch_json,
    progress=lambda s: None,
) -> list[dict]:
    """Fetch one cell via datasets-server /filter: sequential shallow pages into a pool,
    then a SEEDED subsample within that pool.

    Deep random offsets on filtered queries 500 server-side (measured 2026-07-25), so
    randomness lives in the subsample, not the page positions. The window is the first
    ~3x-need matching rows in corpus order — recorded in slice provenance as
    ``sampling: "sequential-window + seeded subsample"`` so no receipt pretends this was
    a uniform draw over the whole cell.
    """
    first = fetch(_filter_url(cell, 0, 1))
    _validate_columns([f["name"] for f in first.get("features", [])])
    total = int(first.get("num_rows_total") or 0)
    if total == 0:
        return []
    pool: dict[str, dict] = {}
    target_pool = need * 3
    max_pages = (target_pool // PAGE) + 4
    for page_i in range(max_pages):
        offset = page_i * PAGE
        if offset >= total:
            break
        page = fetch(_filter_url(cell, offset, PAGE))
        for r in page.get("rows", []):
            m = _map_row(r.get("row", {}))
            if m and m["id"] not in pool and _word_ok(m["text"], min_words, max_words):
                pool[m["id"]] = m
        progress(f"  {cell.key()}: pool {len(pool)} (page {page_i + 1}, n_total={total})")
        if len(pool) >= target_pool:
            break
    rows = list(pool.values())
    if len(rows) <= need:
        return rows
    return rng.sample(rows, need)


# --------------------------------------------------------------------------- csv path


def sample_csv(
    raid_csv: Path,
    cells: list[Cell],
    need: int,
    rng: random.Random,
    *,
    min_words: int,
    max_words: int,
) -> dict[str, list[dict]]:
    """Single streaming pass, per-cell reservoir sampling. Stdlib csv only."""
    # RAID generations exceed the csv module's default 128 KiB field limit.
    csv.field_size_limit(sys.maxsize)
    wanted = {c.key(): c for c in cells}
    reservoirs: dict[str, list[dict]] = {k: [] for k in wanted}
    seen: dict[str, int] = {k: 0 for k in wanted}
    with open(raid_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _validate_columns(reader.fieldnames or [])
        for row in reader:
            key = f"{row.get('model')}|{row.get('domain')}|{row.get('attack')}"
            if key not in wanted:
                continue
            m = _map_row(row)
            if not m or not _word_ok(m["text"], min_words, max_words):
                continue
            seen[key] += 1
            res = reservoirs[key]
            if len(res) < need:
                res.append(m)
            else:
                j = rng.randrange(seen[key])
                if j < need:
                    res[j] = m
    return reservoirs


# --------------------------------------------------------------------------- prepare


def prepare_raid_slice(
    out_dir: Path,
    *,
    generators: list[str],
    domains: list[str],
    attacks: list[str],
    per_cell: int = 60,
    seed: int = 17,
    min_words: int = 50,
    max_words: int = 500,
    source: str = "hub",
    raid_csv: Path | None = None,
    fetch=_fetch_json,
    progress=print,
) -> corpus.Corpus:
    """Prepare a seeded RAID slice and return it as a gaige Corpus.

    Human rows are sampled per domain (model="human", attack="none" — RAID applies
    attacks to machine text). AI cells are the cross product generators × domains ×
    attacks. The slice lands in ``out_dir`` (gitignored ``corpora/`` by convention);
    provenance rides in ``Corpus.meta`` and therefore in env.json verbatim.
    """
    # Validate inputs BEFORE touching the filesystem — a refusal should leave no trace.
    if source == "csv" and (not raid_csv or not Path(raid_csv).exists()):
        raise RaidFetchError(
            "source=csv needs --csv pointing at a downloaded RAID csv "
            "(train.csv is ~11.8 GB from https://huggingface.co/datasets/liamdugan/raid)."
        )
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    ai_cells = [Cell(g, d, a) for g in generators for d in domains for a in attacks if g != "human"]
    human_cells = [Cell("human", d, "none") for d in domains]
    revision = "local-csv" if source == "csv" else _revision_sha(fetch)

    rows: list[dict] = []
    if source == "csv":
        pools = sample_csv(
            Path(raid_csv),
            human_cells + ai_cells,
            per_cell,
            rng,
            min_words=min_words,
            max_words=max_words,
        )
        for key, got in pools.items():
            progress(f"  {key}: {len(got)}/{per_cell}")
            rows.extend(got)
    else:
        for cell in human_cells + ai_cells:
            got = sample_cell_hub(
                cell,
                per_cell,
                rng,
                min_words=min_words,
                max_words=max_words,
                fetch=fetch,
                progress=progress,
            )
            progress(f"  {cell.key()}: {len(got)}/{per_cell}")
            rows.extend(got)

    if not rows:
        raise RaidFetchError("no rows survived selection — loosen cells/filters")
    rng.shuffle(rows)

    sel = f"g{len(generators)}d{len(domains)}a{len(attacks)}-n{per_cell}"
    out_path = out_dir / f"raid-{sel}-s{seed}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = corpus.load_jsonl(out_path)
    c.meta.update(
        {
            "dataset": DATASET,
            "dataset_revision": revision,
            "source": ("datasets-server pages" if source == "hub" else f"local csv {raid_csv}"),
            "selection": {
                "generators": generators,
                "domains": domains,
                "attacks": attacks,
                "per_cell": per_cell,
                "seed": seed,
                "sampling": (
                    "sequential-window + seeded subsample"
                    if source == "hub"
                    else "seeded per-cell reservoir over full csv"
                ),
            },
            "filters": {"min_words": min_words, "max_words": max_words},
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "citation": CITATION,
            "license": LICENSE_NOTE,
            "note": (
                f"RAID slice ({sel}, seed {seed}): thresholds measured here describe "
                "THIS slice's generators/domains/attacks, not RAID, and not deployment."
            ),
        }
    )
    return c
