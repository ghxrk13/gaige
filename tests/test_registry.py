# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Registry tests: identity keys the series, material is frozen per vintage, a changed
instrument FORKS rather than mixes, and the variance bound is measured — all via the fake
provider, no model, no network."""

from __future__ import annotations

import json

import pytest

from gaige import proberun, registry
from gaige.probes import load_probes
from gaige.providers.base import Decoding
from tests.test_proberun import FakeProvider, make_set, right_answers
from tests.test_probes import probe, write_probeset


def do_run(tmp_path, ps, name, decoding=Decoding(), wrong_ids=()):
    out = tmp_path / name
    proberun.run_probes(
        ps,
        FakeProvider(right_answers(ps, wrong_ids=wrong_ids)),
        decoding,
        cutoff="2024-06-01",
        outdir=out,
        n_boot=50,
        progress=lambda *_: None,
    )
    return out


def test_identity_hash_is_order_independent_and_decoding_sensitive():
    inst = {
        "provider": {"provider": "fake", "model_id": "m", "attestation": "verified"},
        "decoding": {"temperature": 0.0, "top_p": 1.0},
        "grading_version": "nem-1",
        "training_cutoff": "2024-06-01",
        "probes_sha256": "ignored-for-identity",
    }
    scrambled = json.loads(json.dumps(inst))  # fresh dicts, potential different key order
    a = registry.series_id(registry.series_identity(inst, "0.0.1"))
    b = registry.series_id(registry.series_identity(scrambled, "0.0.1"))
    assert a == b
    inst2 = json.loads(json.dumps(inst))
    inst2["decoding"]["temperature"] = 0.7
    assert registry.series_id(registry.series_identity(inst2, "0.0.1")) != a
    # probes_sha256 is deliberately NOT identity: the probe file grows with new vintages
    inst3 = json.loads(json.dumps(inst))
    inst3["probes_sha256"] = "different"
    assert registry.series_id(registry.series_identity(inst3, "0.0.1")) == a


def test_same_instrument_grows_one_series(tmp_path):
    ps = make_set(tmp_path)
    reg = tmp_path / "registry"
    s1 = registry.record_run(reg, do_run(tmp_path, ps, "run1"))
    s2 = registry.record_run(reg, do_run(tmp_path, ps, "run2", wrong_ids=("p0",)))
    assert s1["series_id"] == s2["series_id"]
    assert len(s2["runs"]) == 2
    report = (reg / s2["series_id"] / "series-report.md").read_text(encoding="utf-8")
    assert report.count("| 20") == 2  # two run rows, UTC-dated
    assert "Instrument constancy is asserted mechanically" in report


def test_changed_instrument_forks_a_new_series(tmp_path):
    ps = make_set(tmp_path)
    reg = tmp_path / "registry"
    s1 = registry.record_run(reg, do_run(tmp_path, ps, "run1"))
    s2 = registry.record_run(
        reg, do_run(tmp_path, ps, "run2", decoding=Decoding(temperature=0.7, seed=3))
    )
    assert s1["series_id"] != s2["series_id"]  # forked, never mixed
    assert len(registry.list_series(reg)) == 2


def test_edited_vintage_is_refused_by_name(tmp_path):
    ps = make_set(tmp_path)
    reg = tmp_path / "registry"
    registry.record_run(reg, do_run(tmp_path, ps, "run1"))
    rows = [probe(i, vintage="t0") for i in range(4)]
    rows[0]["answer"] = "EDITED"  # same vintage label, different content
    rows += [probe(100 + i, vintage="t1") for i in range(4)]
    edited = load_probes(write_probeset(tmp_path / "edited.jsonl", rows))
    run = do_run(tmp_path, edited, "run2")
    with pytest.raises(registry.SeriesMismatch, match="vintage 't0' content changed"):
        registry.record_run(reg, run)


def test_new_vintage_label_is_welcome(tmp_path):
    ps = make_set(tmp_path)
    reg = tmp_path / "registry"
    registry.record_run(reg, do_run(tmp_path, ps, "run1"))
    rows = [probe(i, vintage="t0") for i in range(4)]
    rows += [probe(100 + i, vintage="t1") for i in range(4)]
    rows += [probe(200 + i, vintage="t2") for i in range(3)]  # the longitudinal design
    grown = load_probes(write_probeset(tmp_path / "grown.jsonl", rows))
    s = registry.record_run(reg, do_run(tmp_path, grown, "run2"))
    assert sorted(s["vintage_hashes"]) == ["t0", "t1", "t2"]


def test_ptrue_template_is_frozen_per_series(tmp_path):
    """M3 riding along does not fork the series, but a changed P(True) template refuses."""
    import json as _json

    from gaige.providers.base import Decoding as _D
    from tests.test_proberun import FakeProvider, right_answers

    ps = make_set(tmp_path)
    reg = tmp_path / "registry"
    out1 = tmp_path / "pt1"
    proberun.run_probes(
        ps,
        FakeProvider(right_answers(ps), conf=(0.8, 0.9)),
        _D(),
        cutoff="2024-06-01",
        outdir=out1,
        n_boot=50,
        with_ptrue=True,
        progress=lambda *_: None,
    )
    s = registry.record_run(reg, out1)
    assert s["ptrue"]["version"] == "ptrue-1"
    registry.record_run(reg, do_run(tmp_path, ps, "plain"))  # ptrue-off run joins, no fork
    assert len(registry.list_series(reg)) == 1

    out2 = tmp_path / "pt2"
    proberun.run_probes(
        ps,
        FakeProvider(right_answers(ps), conf=(0.8, 0.9)),
        _D(),
        cutoff="2024-06-01",
        outdir=out2,
        n_boot=50,
        with_ptrue=True,
        progress=lambda *_: None,
    )
    r = _json.loads((out2 / "probe-results.json").read_text(encoding="utf-8"))
    r["instrument"]["ptrue"]["template_sha256"] = "0" * 64  # simulate a changed template
    (out2 / "probe-results.json").write_text(_json.dumps(r), encoding="utf-8")
    with pytest.raises(registry.SeriesMismatch, match="ptrue template changed"):
        registry.record_run(reg, out2)


def test_replicates_measure_the_bound_and_movement_flags(tmp_path):
    ps = make_set(tmp_path)
    reg = tmp_path / "registry"
    for i in (1, 2, 3):  # deterministic fake -> identical accuracy -> bound 0, measured
        registry.record_run(reg, do_run(tmp_path, ps, f"rep{i}"), replicate=True)
    moved = do_run(tmp_path, ps, "moved", wrong_ids=("p0", "p1"))  # t0 drops to 50%
    s = registry.record_run(reg, moved)
    bound = registry.variance_bound(s)
    assert bound["t0"]["bound"] == 0.0 and bound["t0"]["n_replicates"] == 3
    report = (reg / s["series_id"] / "series-report.md").read_text(encoding="utf-8")
    assert "BEYOND the bound" in report  # t0 moved -50% against a 0-width bound
    assert "within run variance" in report  # t1 did not move
