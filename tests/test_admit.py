# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The admit pipeline, end to end, without a GPU.

A synthetic baseline receipt is built with the real writer (compute_results +
write_report over seeded normals), so admit is exercised against exactly the artifact a
real `gaige run` leaves behind. The live lane injects a fake detector and injected live
versions/device — the score_document precedent — so scoring, crash-safety bookkeeping,
and the instrument-mismatch refusal are all covered on machines that could never load a
model. Every floor is proven by hitting it.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from gaige import admit, analyze, cli, receipts
from gaige.analyze import NotAReport

DET_META = {
    "detector": "fast-detect-gpt",
    "model_id": "test-model",
    "quant_requested": "fp32",
    "quant_verified": {},
    "max_tokens": 64,
    "versions": {"torch": "0.test"},
    "device": "cpu",
    "score_semantics": "higher = more AI-like",
}


class FakeDet:
    """Deterministic stand-in scorer for the live lane."""

    def __init__(self, base: float = 1.0):
        self.base = base

    def score(self, text: str) -> float:
        return self.base + 0.001 * len(text.split())


def _make_baseline(tmp_path, n_h=100, n_a=100, det_meta=None):
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_h):
        rows.append(
            {
                "id": f"h{i}",
                "label": "human",
                "score": float(rng.normal(0.0, 1.0)),
                "seconds": 0.0,
                "n_words": 120,
                "meta": {"domain": "a" if i % 2 else "b"},
            }
        )
    for i in range(n_a):
        rows.append(
            {
                "id": f"a{i}",
                "label": "ai",
                "score": float(rng.normal(2.0, 1.0)),
                "seconds": 0.0,
                "n_words": 120,
                "meta": {"domain": "a" if i % 2 else "b"},
            }
        )
    results = analyze.compute_results(rows, n_boot=50, seed=17)
    outdir = tmp_path / "baseline-report"
    view = analyze.CorpusView(
        name="synth",
        sha256="0" * 64,
        counts={"human": n_h, "ai": n_a},
        meta={"source": "synthetic"},
    )
    receipts.write_report(
        outdir, view, det_meta or DET_META, rows, results, "gaige run --synthetic"
    )
    return outdir


def _write_candidate(tmp_path, n=30, words=60, meta=True, labels=False, name="candidate.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            row = {"id": f"c{i}", "text": " ".join(["word"] * words)}
            if meta:
                row["meta"] = {"domain": "a" if i % 2 else "b"}
            if labels:
                row["label"] = "human"
            f.write(json.dumps(row) + "\n")
    return path


def _write_scores_csv(tmp_path, scores, n_words=None, metas=None, name="cand-scores.csv"):
    path = tmp_path / name
    fields = ["id", "score"]
    if n_words is not None:
        fields.append("n_words")
    if metas is not None:
        fields.append("meta")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, s in enumerate(scores):
            row = {"id": f"c{i}", "score": s}
            if n_words is not None:
                row["n_words"] = n_words[i]
            if metas is not None:
                row["meta"] = json.dumps(metas[i])
            w.writerow(row)
    return path


# ------------------------------------------------------------------ loaders


def test_candidate_loader_defaults_ids_and_notes_labels(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        '{"text": "some words here"}\n{"id": "x1", "text": "more words", "label": "ai"}\n',
        encoding="utf-8",
    )
    s = admit.load_candidate_jsonl(p)
    assert s.n == 2
    assert s.items[0]["id"] == "c-0"
    assert s.items[1]["id"] == "x1"
    assert s.labels_present is True
    assert len(s.sha256) == 64


def test_candidate_loader_refuses_missing_text(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"text": "fine"}\n{"id": "bad"}\n', encoding="utf-8")
    with pytest.raises(ValueError) as e:
        admit.load_candidate_jsonl(p)
    assert ":2:" in str(e.value)


def test_candidate_loader_refuses_empty_file(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        admit.load_candidate_jsonl(p)


def test_scores_reader_minimal_and_missing_column(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("score\n1.0\n2.0\n", encoding="utf-8")
    rows, sha, labels = admit.read_candidate_scores_csv(p)
    assert [r["score"] for r in rows] == [1.0, 2.0]
    assert rows[0]["id"] == "row0"
    assert labels is False
    bad = tmp_path / "bad.csv"
    bad.write_text("id,value\nx,1\n", encoding="utf-8")
    with pytest.raises(NotAReport):
        admit.read_candidate_scores_csv(bad)


def test_scores_reader_flags_label_column(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,score,label\nc0,1.0,human\n", encoding="utf-8")
    _, _, labels = admit.read_candidate_scores_csv(p)
    assert labels is True


# ------------------------------------------------------------------ refusals


def test_baseline_without_env_defines_no_standard(tmp_path):
    baseline = _make_baseline(tmp_path)
    (baseline / "env.json").unlink()
    scores = _write_scores_csv(tmp_path, [1.0] * 25)
    with pytest.raises(ValueError) as e:
        admit.run_admit(baseline, candidate_scores_path=scores, out=tmp_path / "o", n_boot=50)
    assert "defines no standard" in str(e.value)


def test_baseline_without_scores_is_not_a_report(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    scores = _write_scores_csv(tmp_path, [1.0] * 25)
    with pytest.raises(NotAReport):
        admit.run_admit(d, candidate_scores_path=scores, out=tmp_path / "o", n_boot=50)


def test_exactly_one_source_required(tmp_path):
    baseline = _make_baseline(tmp_path)
    with pytest.raises(ValueError):
        admit.run_admit(baseline, out=tmp_path / "o", n_boot=50)


def test_live_lane_refuses_on_instrument_mismatch(tmp_path):
    baseline = _make_baseline(tmp_path)
    cand = _write_candidate(tmp_path, n=25)
    with pytest.raises(RuntimeError) as e:
        admit.run_admit(
            baseline,
            candidate_path=cand,
            out=tmp_path / "o",
            n_boot=50,
            detector=FakeDet(),
            live_versions={"torch": "9.9"},
            live_device="cpu",
        )
    msg = str(e.value)
    assert "refusing the live lane" in msg
    assert "torch" in msg
    assert "--candidate-scores" in msg
    assert not (tmp_path / "o").exists()  # a refusal leaves no trace


def test_live_lane_device_mismatch_refuses(tmp_path):
    baseline = _make_baseline(tmp_path)
    cand = _write_candidate(tmp_path, n=25)
    with pytest.raises(RuntimeError) as e:
        admit.run_admit(
            baseline,
            candidate_path=cand,
            out=tmp_path / "o",
            n_boot=50,
            detector=FakeDet(),
            live_versions={"torch": "0.test"},
            live_device="cuda",
        )
    assert "different numerics" in str(e.value)


def test_live_lane_names_remedy_for_non_fdg_baseline(tmp_path):
    baseline = _make_baseline(tmp_path, det_meta=dict(DET_META, detector="binoculars"))
    cand = _write_candidate(tmp_path, n=25)
    with pytest.raises(ValueError) as e:
        admit.run_admit(
            baseline,
            candidate_path=cand,
            out=tmp_path / "o",
            n_boot=50,
            live_versions={"torch": "0.test"},
            live_device="cpu",
        )
    assert "--candidate-scores" in str(e.value)


def test_bad_reference_mode_refuses(tmp_path):
    baseline = _make_baseline(tmp_path)
    scores = _write_scores_csv(tmp_path, [1.0] * 25)
    with pytest.raises(ValueError):
        admit.run_admit(
            baseline, candidate_scores_path=scores, reference="trusted", out=tmp_path / "o"
        )


# ------------------------------------------------------------------ scores lane, end to end


def test_scores_lane_end_to_end_via_cli(tmp_path, capsys):
    baseline = _make_baseline(tmp_path)
    rng = np.random.default_rng(5)
    scores = [float(x) for x in rng.normal(1.0, 1.0, size=40)]
    p = _write_scores_csv(tmp_path, scores, n_words=[120] * 40, metas=[{"domain": "a"}] * 40)
    out = tmp_path / "admit-out"
    rc = cli.main(
        [
            "admit",
            "--baseline",
            str(baseline),
            "--candidate-scores",
            str(p),
            "--out",
            str(out),
            "--n-boot",
            "50",
        ]
    )
    captured = capsys.readouterr().out
    assert rc == 0
    assert "WARNING" in captured and "attestation" in captured
    assert "never a verdict" in captured
    for name in ("report.md", "results.json", "env.json", "candidate-scores.csv"):
        assert (out / name).exists()
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["kind"] == "admit"
    assert res["candidate"]["scoring"] == "supplied-unattested"
    assert res["reference"] == {"mode": "all", "n": 200, "n_human": 100, "n_ai": 100}
    # pooled n=200: alpha .05 and .01 supported, .005 refused (needs 399)
    by_alpha = {r["alpha"]: r for r in res["novelty"]}
    assert "unavailable" not in by_alpha[0.05]
    assert "unavailable" not in by_alpha[0.01]
    assert "unavailable" in by_alpha[0.005]
    assert res["primary_alpha"] == 0.05
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "UNATTESTED" in report
    assert "never says admit or reject" in report
    assert "no roc" not in report  # sanity: nothing pretends a ROC exists
    assert not (out / "roc.json").exists()


def test_reference_human_refuses_alpha_001(tmp_path):
    baseline = _make_baseline(tmp_path)
    scores = _write_scores_csv(tmp_path, [float(x) for x in np.linspace(-1, 2, 30)])
    r = admit.run_admit(
        baseline,
        candidate_scores_path=scores,
        reference="human",
        out=tmp_path / "o",
        n_boot=50,
    )
    res = r["results"]
    assert res["reference"]["mode"] == "human"
    assert res["reference"]["n"] == 100
    by_alpha = {row["alpha"]: row for row in res["novelty"]}
    assert "unavailable" not in by_alpha[0.05]  # needs 39
    assert "unavailable" in by_alpha[0.01]  # needs 199 > 100
    assert res["primary_alpha"] == 0.05


def test_small_candidate_withholds_slice_stats_but_places_documents(tmp_path):
    baseline = _make_baseline(tmp_path)
    scores = _write_scores_csv(tmp_path, [1.0] * 10)
    out = tmp_path / "o"
    r = admit.run_admit(baseline, candidate_scores_path=scores, out=out, n_boot=50)
    res = r["results"]
    assert res["floors"]["slice_stats_withheld"] is True
    assert res["novelty"] is None and res["ks"] is None and res["quantile_shift"] is None
    with open(out / "candidate-scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert all(r["percentile_ref"] not in ("", None) for r in rows)
    assert all(r["conformal_p_two_sided"] not in ("", None) for r in rows)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "below" in report and "floor" in report


def test_percentile_extremes(tmp_path):
    baseline = _make_baseline(tmp_path)
    vals = [-100.0] + [1.0] * 18 + [100.0]
    scores = _write_scores_csv(tmp_path, vals)
    out = tmp_path / "o"
    admit.run_admit(baseline, candidate_scores_path=scores, out=out, n_boot=50)
    with open(out / "candidate-scores.csv", newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert float(rows["c0"]["percentile_ref"]) == 0.0
    assert float(rows["c19"]["percentile_ref"]) == 1.0


def test_strata_report_and_withholding(tmp_path):
    baseline = _make_baseline(tmp_path)
    n_a, n_b = 25, 10
    scores = [1.0] * (n_a + n_b)
    metas = [{"domain": "a"}] * n_a + [{"domain": "b"}] * n_b
    p = _write_scores_csv(tmp_path, scores, n_words=[120] * (n_a + n_b), metas=metas)
    r = admit.run_admit(baseline, candidate_scores_path=p, out=tmp_path / "o", n_boot=50)
    strata = r["results"]["strata"]
    dom = strata["axes"]["domain"]
    assert dom["a"]["n"] == n_a and dom["a"]["rate_withheld"] is False
    assert "outside_rate" in dom["a"] and "median_percentile_ref" in dom["a"]
    assert dom["b"]["n"] == n_b and dom["b"]["rate_withheld"] is True
    assert "outside_rate" not in dom["b"]
    # baseline context: the synthetic baseline carries both axes on every row
    assert dom["a"]["n_baseline"] == 100  # 50 human + 50 ai carry domain=a
    assert strata["axes"]["length_bucket"]["100-250w"]["n_baseline"] == 200


def test_determinism_same_inputs_same_bytes(tmp_path):
    baseline = _make_baseline(tmp_path)
    scores = _write_scores_csv(tmp_path, [float(x) for x in np.linspace(-1, 3, 40)])
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    admit.run_admit(baseline, candidate_scores_path=scores, out=out1, n_boot=50)
    admit.run_admit(baseline, candidate_scores_path=scores, out=out2, n_boot=50)
    assert (out1 / "results.json").read_bytes() == (out2 / "results.json").read_bytes()
    assert (out1 / "candidate-scores.csv").read_bytes() == (
        out2 / "candidate-scores.csv"
    ).read_bytes()


# ------------------------------------------------------------------ live lane


def test_live_lane_with_injected_detector(tmp_path, capsys):
    baseline = _make_baseline(tmp_path)
    cand = _write_candidate(tmp_path, n=25, words=60, labels=True)
    out = tmp_path / "live-out"
    r = admit.run_admit(
        baseline,
        candidate_path=cand,
        out=out,
        n_boot=50,
        detector=FakeDet(base=1.5),
        live_versions={"torch": "0.test"},
        live_device="cpu",
    )
    res = r["results"]
    assert res["candidate"]["scoring"] == "live"
    assert res["candidate"]["labels_ignored"] is True
    assert res["candidate"]["n"] == 25
    assert "labels on candidate rows are ignored" in capsys.readouterr().out
    # crash-safety bookkeeping: runstate written, partial cleaned up on completion
    assert (out / "run.json").exists()
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["complete"] is True
    assert not (out / "scores.partial.csv").exists()
    # every score is the fake detector's arithmetic: base 1.5 + 0.001 * 60 words
    with open(out / "candidate-scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(float(row["score"]) == pytest.approx(1.56) for row in rows)
    assert all(row["short_text"] == "False" for row in rows)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "UNATTESTED" not in report
    assert "live" in report


def test_live_lane_counts_short_texts(tmp_path):
    baseline = _make_baseline(tmp_path)
    cand = _write_candidate(tmp_path, n=25, words=30)  # every doc under the 50-word floor
    out = tmp_path / "o"
    r = admit.run_admit(
        baseline,
        candidate_path=cand,
        out=out,
        n_boot=50,
        detector=FakeDet(),
        live_versions={"torch": "0.test"},
        live_device="cpu",
    )
    res = r["results"]
    assert res["candidate"]["n_short_text"] == 25
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "under 50 words" in report
