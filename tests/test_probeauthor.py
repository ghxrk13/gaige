# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

import hashlib
import json

import pytest

from gaige import probeauthor
from gaige.cli import main
from gaige.providers.base import Decoding

CUTOFF = "2026-01-01"
SHA_OK = "ab" * 32


def write_probeset(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def probe(i, vintage="t0", source_date="2026-01-15", authored="2026-01-20", **kw):
    return {
        "id": f"p{i}",
        "prompt": f"Question {i}?",
        "answer": f"answer{i}",
        "vintage": vintage,
        "source": "unit-test fixture source",
        "source_date": source_date,
        "authored": authored,
        **kw,
    }


def manifest(filename, **over):
    m = {
        "manifest_version": 1,
        "probe_file": over.pop("probe_file", filename),
        "training_cutoff": CUTOFF,
        "grading": {"rule": "normalized-exact-match", "version": "nem-1"},
        "decoding": {"policy": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 0},
        "control": {
            "benchmark": "mmlu-subset-v1 (frozen fixture)",
            "sha256": SHA_OK,
            "path": None,
            "scoring": "option-logprob-argmax",
        },
    }
    m.update(over)
    return m


def valid_set(tmp_path, rows=None, manifest_over=None):
    p = write_probeset(tmp_path / "set.jsonl", rows or [probe(1), probe(2)])
    m = manifest("set.jsonl", **(manifest_over or {}))
    probeauthor.manifest_path(p).write_text(json.dumps(m), encoding="utf-8")
    return p


# --- template generator ---


def test_new_scaffolds_template_and_manifest(tmp_path):
    out, mpath = probeauthor.new_probe_set(tmp_path / "t0.jsonl", vintage="t0", cutoff=CUTOFF)
    assert out.exists() and mpath.exists()
    m = json.loads(mpath.read_text(encoding="utf-8"))
    # The fixed decisions arrive pre-filled and correct.
    assert m["grading"] == {"rule": "normalized-exact-match", "version": "nem-1"}
    assert m["decoding"]["policy"] == "greedy" and m["decoding"]["temperature"] == 0.0
    assert m["control"]["scoring"] == "option-logprob-argmax"
    assert m["training_cutoff"] == CUTOFF
    # Template rows are schema-valid JSONL (loadable), dated just past the cutoff.
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert all(r["source_date"] == "2026-01-02" for r in rows)
    assert all("authored" in r for r in rows)


def test_new_refuses_overwrite(tmp_path):
    probeauthor.new_probe_set(tmp_path / "t0.jsonl", cutoff=CUTOFF)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        probeauthor.new_probe_set(tmp_path / "t0.jsonl", cutoff=CUTOFF)


def test_new_rejects_bad_cutoff(tmp_path):
    with pytest.raises(ValueError, match="ISO date"):
        probeauthor.new_probe_set(tmp_path / "t0.jsonl", cutoff="mid 2026")


def test_template_fails_lint_until_authored(tmp_path):
    out, _ = probeauthor.new_probe_set(tmp_path / "t0.jsonl", cutoff=CUTOFF)
    rep = probeauthor.lint(out)
    assert not rep.ok
    assert any("placeholder" in e for e in rep.errors)
    # Both the rows and the manifest's control linkage still carry placeholders.
    assert any(e.startswith("probe ") for e in rep.errors)
    assert any("control" in e for e in rep.errors)


# --- lint: the signed decisions have teeth ---


def test_authored_set_passes_lint(tmp_path):
    rep = probeauthor.lint(valid_set(tmp_path))
    assert rep.ok and rep.warnings == []
    assert rep.vintages == {"t0": {"n": 2, "post_cutoff": 2}}


def test_missing_manifest_is_error(tmp_path):
    p = write_probeset(tmp_path / "set.jsonl", [probe(1)])
    rep = probeauthor.lint(p)
    assert any("manifest" in e and "missing" in e for e in rep.errors)


def test_wrong_grading_version_rejected(tmp_path):
    p = valid_set(
        tmp_path,
        manifest_over={"grading": {"rule": "normalized-exact-match", "version": "nem-0"}},
    )
    rep = probeauthor.lint(p)
    assert any("grading.version" in e for e in rep.errors)


def test_non_greedy_declaration_rejected(tmp_path):
    p = valid_set(tmp_path, manifest_over={"decoding": {"policy": "sampling", "temperature": 0.7}})
    rep = probeauthor.lint(p)
    assert any("greedy" in e for e in rep.errors)


def test_control_linkage_required(tmp_path):
    c = {"benchmark": "", "sha256": None, "scoring": "llm-judge"}
    rep = probeauthor.lint(valid_set(tmp_path, manifest_over={"control": c}))
    assert any("control.scoring" in e for e in rep.errors)
    assert any("control.benchmark" in e for e in rep.errors)
    assert any("control.sha256" in e for e in rep.errors)


def test_control_hash_verified_when_file_present(tmp_path):
    control = tmp_path / "control.jsonl"
    control.write_text('{"q": 1}\n', encoding="utf-8")
    good = hashlib.sha256(control.read_bytes()).hexdigest()
    c_ok = {
        "benchmark": "fixture",
        "sha256": good,
        "path": "control.jsonl",
        "scoring": "option-logprob-argmax",
    }
    assert probeauthor.lint(valid_set(tmp_path, manifest_over={"control": c_ok})).ok
    c_bad = dict(c_ok, sha256=SHA_OK)
    rep = probeauthor.lint(valid_set(tmp_path, manifest_over={"control": c_bad}))
    assert any("control" in e and "declares" in e for e in rep.errors)


def test_missing_authored_field_is_error(tmp_path):
    r = probe(1)
    del r["authored"]
    rep = probeauthor.lint(valid_set(tmp_path, rows=[r]))
    assert any("authored" in e and "decision e" in e for e in rep.errors)


def test_pre_cutoff_source_date_is_error(tmp_path):
    rows = [
        probe(1, source_date=CUTOFF, authored="2026-01-20"),
        probe(2, source_date="2025-06-01", authored="2026-01-20"),
    ]
    rep = probeauthor.lint(valid_set(tmp_path, rows=rows))
    hits = [e for e in rep.errors if "post-date the" in e]
    assert len(hits) == 2  # the cutoff day itself does not post-date the cutoff


def test_authored_before_source_is_error(tmp_path):
    rep = probeauthor.lint(
        valid_set(tmp_path, rows=[probe(1, source_date="2026-01-15", authored="2026-01-10")])
    )
    assert any("cannot be authored before its source" in e for e in rep.errors)


def test_ungradeable_answer_is_error(tmp_path):
    rep = probeauthor.lint(valid_set(tmp_path, rows=[probe(1, answer="???")]))
    assert any("normalizes to nothing" in e for e in rep.errors)


def test_long_answer_warns_not_fails(tmp_path):
    r = probe(1, answer="a rather long answer of very many words")
    rep = probeauthor.lint(valid_set(tmp_path, rows=[r]))
    assert rep.ok
    assert any("short checkable answers" in w for w in rep.warnings)


def test_redundant_alias_warns(tmp_path):
    rep = probeauthor.lint(
        valid_set(tmp_path, rows=[probe(1, answer="Paris", aliases=["the paris"])])
    )
    assert rep.ok
    assert any("duplicates another key" in w for w in rep.warnings)


def test_duplicate_prompt_warns(tmp_path):
    r1, r2 = probe(1), probe(2)
    r2["prompt"] = r1["prompt"]
    rep = probeauthor.lint(valid_set(tmp_path, rows=[r1, r2]))
    assert any("prompt duplicates" in w for w in rep.warnings)


def test_manifest_probe_file_mismatch_is_error(tmp_path):
    rep = probeauthor.lint(valid_set(tmp_path, manifest_over={"probe_file": "other.jsonl"}))
    assert any("binds exactly one probe file" in e for e in rep.errors)


def test_bad_jsonl_surfaces_loader_error(tmp_path):
    p = tmp_path / "set.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    probeauthor.manifest_path(p).write_text(json.dumps(manifest("set.jsonl")), encoding="utf-8")
    rep = probeauthor.lint(p)
    assert any("not valid JSON" in e for e in rep.errors)


# --- run-time enforcement (decision d is mechanical, not editorial) ---


def test_run_without_manifest_is_unenforced_note(tmp_path):
    p = write_probeset(tmp_path / "legacy.jsonl", [probe(1)])
    note = probeauthor.check_run_against_manifest(p, Decoding(temperature=0.0))
    assert "unenforced" in note


def test_run_refuses_non_greedy_against_manifest(tmp_path):
    p = valid_set(tmp_path)
    with pytest.raises(probeauthor.ManifestViolation, match="pre-registers greedy"):
        probeauthor.check_run_against_manifest(p, Decoding(temperature=0.3))
    note = probeauthor.check_run_against_manifest(p, Decoding(temperature=0.0))
    assert "manifest honored" in note


def test_run_refuses_lint_errors(tmp_path):
    r = probe(1)
    del r["authored"]
    p = valid_set(tmp_path, rows=[r])
    with pytest.raises(probeauthor.ManifestViolation, match="fails its own manifest lint"):
        probeauthor.check_run_against_manifest(p, Decoding(temperature=0.0))


# --- CLI wiring ---


def test_cli_new_then_lint_roundtrip(tmp_path, capsys):
    out = tmp_path / "t0.jsonl"
    assert main(["probe", "new", "--out", str(out), "--cutoff", CUTOFF]) == 0
    assert main(["probe", "lint", "--probes", str(out)]) == 1  # template must not pass
    # Author it: real probes, real control linkage.
    write_probeset(out, [probe(1), probe(2)])
    mp = probeauthor.manifest_path(out)
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["control"] = {
        "benchmark": "mmlu-subset-v1 (frozen fixture)",
        "sha256": SHA_OK,
        "path": None,
        "scoring": "option-logprob-argmax",
    }
    mp.write_text(json.dumps(m), encoding="utf-8")
    assert main(["probe", "lint", "--probes", str(out)]) == 0
    assert "[lint] PASS" in capsys.readouterr().out


def test_cli_new_refuses_overwrite(tmp_path):
    out = tmp_path / "t0.jsonl"
    assert main(["probe", "new", "--out", str(out), "--cutoff", CUTOFF]) == 0
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["probe", "new", "--out", str(out), "--cutoff", CUTOFF])
