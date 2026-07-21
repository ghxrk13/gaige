"""Corpus loading. Format: JSONL rows {"id": str, "text": str, "label": "human"|"ai"}.

Built-in fetcher: hc3-mini — a seeded subsample of HC3 (Hello-SimpleAI), the classic
human-vs-ChatGPT answer corpus. Public, ungated, cached locally, fingerprinted by sha256.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import requests

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"


@dataclass
class Corpus:
    name: str
    path: Path
    items: list[dict]
    sha256: str
    meta: dict

    @property
    def counts(self) -> dict:
        c: dict = {}
        for it in self.items:
            c[it["label"]] = c.get(it["label"], 0) + 1
        return c


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path, name: str | None = None) -> Corpus:
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") not in ("human", "ai") or not row.get("text"):
                raise ValueError(f"{path}:{i + 1}: rows need text + label in {{human,ai}}")
            row.setdefault("id", f"{path.stem}-{i}")
            items.append(row)
    return Corpus(
        name=name or path.stem,
        path=path,
        items=items,
        sha256=_sha256(path),
        meta={"source": str(path)},
    )


def _clip_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def fetch_hc3_mini(
    corpora_dir: Path,
    n_per_class: int = 100,
    seed: int = 17,
    min_words: int = 50,
    max_words: int = 300,
) -> Corpus:
    """Download HC3 once (cached), then emit a seeded, length-filtered subsample."""
    corpora_dir.mkdir(parents=True, exist_ok=True)
    raw = corpora_dir / "hc3-all.jsonl"
    if not raw.exists():
        tmp = raw.with_suffix(".part")
        with requests.get(HC3_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.rename(raw)

    out = corpora_dir / f"hc3-mini-n{n_per_class}-s{seed}.jsonl"
    if not out.exists():
        humans: list[str] = []
        ais: list[str] = []
        with open(raw, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for t in row.get("human_answers") or []:
                    if len(t.split()) >= min_words:
                        humans.append(_clip_words(t, max_words))
                for t in row.get("chatgpt_answers") or []:
                    if len(t.split()) >= min_words:
                        ais.append(_clip_words(t, max_words))
        rng = random.Random(seed)
        rng.shuffle(humans)
        rng.shuffle(ais)
        if len(humans) < n_per_class or len(ais) < n_per_class:
            raise ValueError(
                f"HC3 after length filter has human={len(humans)} ai={len(ais)}; "
                f"asked for {n_per_class}/class"
            )
        with open(out, "w", encoding="utf-8") as f:
            for i, t in enumerate(humans[:n_per_class]):
                f.write(json.dumps({"id": f"hc3-h{i}", "text": t, "label": "human"}) + "\n")
            for i, t in enumerate(ais[:n_per_class]):
                f.write(json.dumps({"id": f"hc3-a{i}", "text": t, "label": "ai"}) + "\n")

    c = load_jsonl(out, name=f"hc3-mini(n={n_per_class},seed={seed})")
    c.meta.update(
        {
            "source": "HC3 (Hello-SimpleAI) via HF hub",
            "url": HC3_URL,
            "raw_sha256": _sha256(raw),
            "filters": {"min_words": min_words, "max_words": max_words},
            "n_per_class": n_per_class,
            "seed": seed,
            "note": "known-AI side is ChatGPT-era text; detectors may score newer model families differently",
        }
    )
    return c
