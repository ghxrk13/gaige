# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

import json

import pytest

from gaige import probes


def write_probeset(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def probe(i, vintage="t0", source_date="2026-01-15", **kw):
    return {
        "id": f"p{i}",
        "prompt": f"Question {i}?",
        "answer": f"answer {i}",
        "vintage": vintage,
        "source": "unit-test fixture",
        "source_date": source_date,
        **kw,
    }


def test_load_valid_set(tmp_path):
    p = write_probeset(
        tmp_path / "ok.jsonl",
        [probe(1), probe(2, vintage="t1"), probe(3, vintage="t1", aliases=["alt"])],
    )
    ps = probes.load_probes(p)
    assert ps.vintages == {"t0": 1, "t1": 2}
    assert ps.counts == ps.vintages  # the runstate-compatibility alias
    assert len(ps.sha256) == 64


def test_missing_field_names_row_and_remedy(tmp_path):
    rows = [probe(1)]
    del rows[0]["source_date"]
    p = write_probeset(tmp_path / "bad.jsonl", rows)
    with pytest.raises(probes.BadProbeSet, match=r"bad.jsonl:1.*source_date"):
        probes.load_probes(p)


def test_duplicate_id_rejected(tmp_path):
    p = write_probeset(tmp_path / "dup.jsonl", [probe(1), probe(1)])
    with pytest.raises(probes.BadProbeSet, match="duplicate id"):
        probes.load_probes(p)


def test_bad_date_rejected(tmp_path):
    p = write_probeset(tmp_path / "date.jsonl", [probe(1, source_date="Jan 2026")])
    with pytest.raises(probes.BadProbeSet, match="ISO date"):
        probes.load_probes(p)


def test_bad_aliases_rejected(tmp_path):
    p = write_probeset(tmp_path / "al.jsonl", [probe(1, aliases="not-a-list")])
    with pytest.raises(probes.BadProbeSet, match="aliases must be a list"):
        probes.load_probes(p)


def test_empty_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(probes.BadProbeSet, match="no probes"):
        probes.load_probes(p)


def test_post_cutoff_share_arithmetic(tmp_path):
    p = write_probeset(
        tmp_path / "cut.jsonl",
        [
            probe(1, source_date="2024-01-01"),  # pre-cutoff
            probe(2, source_date="2025-01-01"),  # post
            probe(3, vintage="t1", source_date="2025-06-01"),  # post
        ],
    )
    ps = probes.load_probes(p)
    share = ps.post_cutoff_share("2024-06-01")
    assert share["t0"] == {"n": 2, "post_cutoff": 1, "share": 0.5}
    assert share["t1"] == {"n": 1, "post_cutoff": 1, "share": 1.0}
