# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.

"""The export surface: public bytes, held to the public bar.

An exported receipt is prose and numbers on gaige.dev the moment it lands, so these tests
pin the schema string, prove the projection contract in both directions, prove redaction
fails closed on synthetic canaries (never real terms; fixtures ship in the sdist), pin the
exact bytes with a golden file, and hold the golden to the same blocked-claims scan as
every shipped document. A bench machine holding real reports additionally sweeps every
exportable receipt against a local, untracked term list.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from gaige import cli, export
from gaige.analyze import NotAReport

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CLEAN = FIXTURES / "export-clean"
OLDER = FIXTURES / "export-older"
DIRTY = FIXTURES / "export-dirty"
GOLDEN = FIXTURES / "export-golden"


def _results(fixture: Path) -> dict:
    return json.loads((fixture / "results.json").read_text(encoding="utf-8"))


def test_schema_strings_are_pinned():
    doc = export.build_export(CLEAN)
    assert doc["schema"] == "gaige-receipt-export/1"
    assert export.SCHEMA == "gaige-receipt-export/1"
    assert export.INDEX_SCHEMA == "gaige-export-index/1"


def test_projection_contract_both_directions(tmp_path):
    """Every emitted key is enumerated; every statistic lands verbatim; nothing else leaks in."""
    doc = export.build_export(CLEAN)
    assert tuple(doc.keys()) == export.TOP_KEYS  # clean fixture carries every section

    results = _results(CLEAN)
    assert doc["metrics"]["auroc"] == results["auroc"]
    assert doc["metrics"]["auroc_ci"] == results["auroc_ci"]
    assert doc["metrics"]["eer"] == results["eer"]
    assert doc["metrics"]["n_boot"] == results["n_boot"]
    assert doc["thresholds"] == results["thresholds"]
    assert doc["conformal"] == results["conformal"]
    assert doc["subgroups"] == results["subgroups"]
    assert doc["base_rate"] == results["base_rate"]

    env = json.loads((CLEAN / "env.json").read_text(encoding="utf-8"))
    assert doc["instrument"]["detector"] == env["detector"]
    assert doc["corpus"] == env["corpus"]
    assert doc["reproduce"]["run"] == env["reproduce"]
    assert doc["receipt"]["id"] == "export-clean"


def test_host_projection_drops_the_kernel_string():
    doc = export.build_export(CLEAN)
    assert doc["instrument"]["host"] == {"os": "Linux", "arch": "x86_64", "device": "cuda"}
    assert "canarykernel" not in json.dumps(doc)


def test_redaction_fails_closed_and_names_the_field(tmp_path):
    with pytest.raises(NotAReport) as e:
        export.build_export(DIRTY)
    msg = str(e.value)
    assert "refusing to export" in msg
    assert "instrument.detector.model_path" in msg
    # A refused export writes nothing, even with force.
    with pytest.raises(NotAReport):
        export.write_export(DIRTY, tmp_path, force=True)
    assert not (tmp_path / "receipts").exists()


def test_redaction_catches_ips_and_foreign_urls(tmp_path):
    """Each scanner class has teeth on its own, not only in the dirty fixture's mix."""
    for bad, needle in [
        ("http://10.0.0.5:1234/v1", "non-public host"),
        ("https://example.com/data.jsonl", "non-public host"),
        ("served from 192.168.1.7 overnight", "IP address"),
        ("C:\\Users\\someone\\models", "absolute path"),
        ("~/models/x.gguf", "absolute path"),
    ]:
        with pytest.raises(NotAReport) as e:
            export._scan_strings({"detector": {"field": bad}}, "")
        assert needle in str(e.value), bad
    # And the shapes receipts legitimately carry pass.
    export._scan_strings(
        {
            "model_id": "tiiuae/falcon-7b + tiiuae/falcon-7b-instruct",
            "source": "corpora/raid-g1d1a1-n5-s1.jsonl",
            "url": "https://huggingface.co/datasets/example/x/resolve/main/all.jsonl",
            "paper": "Hans et al., Binoculars (2024), arXiv:2401.12070",
            "torch": "2.13.0+cu130",
        },
        "",
    )


def test_instrument_unknown_refuses(tmp_path, capsys):
    bare = tmp_path / "bare-report"
    bare.mkdir()
    (bare / "results.json").write_text(_dumps_min(), encoding="utf-8")
    rc = cli.main(["export", "--report", str(bare), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "INSTRUMENT UNKNOWN" in err
    assert "Traceback" not in err


def _dumps_min() -> str:
    return json.dumps({"gaige_version": "0.0.2", "auroc": 0.9, "auroc_ci": [0.8, 1.0]})


def test_non_calibration_receipt_refuses(tmp_path):
    probe_like = tmp_path / "probe-report"
    probe_like.mkdir()
    (probe_like / "results.json").write_text(json.dumps({"by_vintage": {}}), encoding="utf-8")
    (probe_like / "env.json").write_text(json.dumps({"detector": {}}), encoding="utf-8")
    with pytest.raises(NotAReport, match="calibration"):
        export.build_export(probe_like)


def test_determinism_lf_newlines_and_idempotence(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    p1, wrote1 = export.write_export(CLEAN, a)
    p2, _ = export.write_export(CLEAN, b)
    assert wrote1 is True
    assert p1.read_bytes() == p2.read_bytes()
    assert b"\r\n" not in p1.read_bytes()
    # Re-export over identical bytes is a silent no-op; the index rebuild is stable.
    _, wrote_again = export.write_export(CLEAN, a)
    assert wrote_again is False
    i1 = export.rebuild_index(a).read_bytes()
    i2 = export.rebuild_index(a).read_bytes()
    assert i1 == i2
    assert b"\r\n" not in i1


def test_force_semantics(tmp_path):
    path, _ = export.write_export(CLEAN, tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(NotAReport, match="different bytes"):
        export.write_export(CLEAN, tmp_path)
    _, wrote = export.write_export(CLEAN, tmp_path, force=True)
    assert wrote is True


def test_index_contents_and_order(tmp_path):
    export.write_export(CLEAN, tmp_path)
    export.write_export(OLDER, tmp_path)
    (tmp_path / "receipts" / "not-an-export.json").write_text("{}", encoding="utf-8")
    index = json.loads(export.rebuild_index(tmp_path).read_text(encoding="utf-8"))
    assert index["schema"] == "gaige-export-index/1"
    ids = [r["id"] for r in index["receipts"]]
    assert ids == sorted(ids) == ["export-clean", "export-older"]
    clean_row = index["receipts"][0]
    assert clean_row["path"] == "receipts/export-clean.json"
    assert clean_row["detector"] == "binoculars"
    assert clean_row["corpus"] == "testcorp-mini(n=5,seed=1)"
    assert clean_row["auroc"] == 0.9123
    assert len(clean_row["corpus_sha256"]) == 64


def test_older_receipt_shape_absent_sections_are_omitted():
    doc = export.build_export(OLDER)
    assert "eer" not in doc["metrics"] and "eer_threshold" not in doc["metrics"]
    assert "conformal" not in doc and "base_rate" not in doc
    assert doc["subgroups"] == _results(OLDER)["subgroups"]  # the refusal is carried verbatim
    assert set(doc.keys()) <= set(export.TOP_KEYS)
    note = doc["reproduce"]["corpus_note"]
    assert "not redistributed" in note and "prepare-raid" in note
    assert "\u2014" not in note  # public prose carries no em dashes


def test_golden_bytes(tmp_path):
    """The format's documentation is a pinned artifact, not a memory."""
    export.write_export(CLEAN, tmp_path)
    export.rebuild_index(tmp_path)
    got_receipt = (tmp_path / "receipts" / "export-clean.json").read_bytes()
    got_index = (tmp_path / "index.json").read_bytes()
    assert got_receipt == (GOLDEN / "receipts" / "export-clean.json").read_bytes()
    assert got_index == (GOLDEN / "index.json").read_bytes()


def test_golden_passes_the_blocked_claims_scan():
    """Exported JSON is shipped prose the moment it reaches the site; same bar, same list."""
    spec = importlib.util.spec_from_file_location(
        "claims_policy", Path(__file__).resolve().parent / "test_claims_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = (GOLDEN / "receipts" / "export-clean.json").read_text(encoding="utf-8")
    for pattern, why in mod.BLOCKED:
        assert not re.search(pattern, text, re.IGNORECASE | re.DOTALL), why


def test_cli_surface(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["export", "--report", str(CLEAN)])  # --out is required, deliberately
    capsys.readouterr()
    rc = cli.main(["export", "--report", str(CLEAN), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[export] wrote" in out and "index" in out and "binoculars" in out
    rc = cli.main(["export", "--report", str(CLEAN), "--out", str(tmp_path)])
    assert rc == 0
    assert "[export] unchanged" in capsys.readouterr().out


def test_admit_receipts_refuse_export_with_a_typed_message(tmp_path):
    """Admission receipts are a different document type; the refusal names the plan."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"kind": "admit"}), encoding="utf-8")
    (d / "env.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(NotAReport) as e:
        export.build_export(d)
    assert "gaige-admit-export/1" in str(e.value)


REPORTS = Path(__file__).resolve().parent.parent / "reports"
LOCAL_TERMS = Path.home() / ".gaige-leak-terms"


@pytest.mark.skipif(
    not REPORTS.is_dir() or not LOCAL_TERMS.exists(),
    reason="needs real reports/ and the machine-local ~/.gaige-leak-terms (never shipped)",
)
def test_real_reports_export_clean_of_local_terms():
    """On the bench machine only: every exportable real receipt sweeps clean of local terms.

    The term list lives outside the repo on purpose; shipping it would be the leak. Reports
    the exporter refuses (probe receipts, structural hits) are the refusal doing its job.
    """
    patterns = [
        re.compile(line.strip(), re.IGNORECASE)
        for line in LOCAL_TERMS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    hits = []
    for rd in sorted(p for p in REPORTS.iterdir() if p.is_dir()):
        if not (rd / "env.json").exists() or not (rd / "results.json").exists():
            continue
        try:
            doc = export.build_export(rd)
        except NotAReport:
            continue
        text = json.dumps(doc)
        hits += [
            f"{rd.name}: {p.pattern!r} -> {p.search(text).group(0)!r}"
            for p in patterns
            if p.search(text)
        ]
    assert not hits, "machine-local terms inside exportable receipts:\n" + "\n".join(hits)
