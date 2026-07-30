# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The provenance sweep's honesty rules, enforced rather than remembered.

The one that matters most: ABSENT may only be emitted when the carrier proved — via a
per-image probe round trip — that it could have held the mark. Everything else here guards
the closed status vocabulary, the mandatory negative_meaning, and the degrade-loudly paths
(missing optional deps, crashed checks). All logic is exercised through injectable codecs
and readers, so the suite runs on the numpy+requests core with no imaging stack installed.
"""

from __future__ import annotations

import json

import pytest

from gaige import provenance as prov
from gaige.cli import main


class FakeCodec:
    method = "fake"

    def __init__(self, stored: bytes | None = None, carrier_ok: bool = True, boom: bool = False):
        self.stored = stored
        self.carrier_ok = carrier_ok
        self.boom = boom

    def decode(self, path, n_bits: int) -> bytes:
        if self.boom:
            raise RuntimeError("decode exploded")
        if self.stored is not None and len(self.stored) * 8 == n_bits:
            return self.stored
        return b"\x00" * (n_bits // 8)

    def roundtrip(self, path, payload: bytes) -> bool:
        if self.boom:
            raise RuntimeError("roundtrip exploded")
        return self.carrier_ok


class FakeReader:
    def __init__(self, manifest: str, state: str = "Valid"):
        self._manifest, self._state = manifest, state

    def json(self) -> str:
        return self._manifest

    def get_validation_state(self) -> str:
        return self._state


# ------------------------------------------------------------------- the status contract


def test_statuses_are_a_closed_set():
    with pytest.raises(ValueError, match="unknown status"):
        prov.SchemeResult("x", "MAYBE", "…", "…")


def test_every_result_must_state_its_negative_meaning():
    with pytest.raises(ValueError, match="negative"):
        prov.SchemeResult("x", prov.ABSENT, "…", "")


def test_absent_with_failed_selftest_is_structurally_impossible():
    """The overclaim the module exists to prevent cannot even be constructed."""
    with pytest.raises(ValueError, match="INCONCLUSIVE"):
        prov.SchemeResult("x", prov.ABSENT, "…", "…", self_test="fail")


def test_carrier_negative_is_absent_only_when_the_probe_survives():
    ok = prov.carrier_negative("wm", probe_survived=True)
    assert (ok.status, ok.self_test) == (prov.ABSENT, "pass")
    bad = prov.carrier_negative("wm", probe_survived=False)
    assert (bad.status, bad.self_test) == (prov.INCONCLUSIVE, "fail")
    assert "no information" in bad.negative_meaning


def test_probe_payload_matches_the_longest_sought_payload():
    """Recoverability degrades with payload length; the probe must certify the hardest case."""
    assert len(prov.PROBE_PAYLOAD) == max(len(p) for p in prov.KNOWN_IMAGE_PAYLOADS.values())


# ------------------------------------------------------------------- dwtDct image checks


def test_watermark_found_when_a_known_payload_decodes(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    r = prov.check_image_watermark(img, codec=FakeCodec(stored=b"StableDiffusionV1"))
    assert r.status == prov.FOUND
    assert r.evidence["payload"] == "StableDiffusionV1"


def test_watermark_absent_requires_a_surviving_probe(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    r = prov.check_image_watermark(img, codec=FakeCodec(carrier_ok=True))
    assert (r.status, r.self_test) == (prov.ABSENT, "pass")


def test_watermark_inconclusive_when_the_carrier_fails_its_selftest(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    r = prov.check_image_watermark(img, codec=FakeCodec(carrier_ok=False))
    assert (r.status, r.self_test) == (prov.INCONCLUSIVE, "fail")


def test_watermark_crash_degrades_to_a_loud_error(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    r = prov.check_image_watermark(img, codec=FakeCodec(boom=True))
    assert r.status == prov.ERROR
    assert "RuntimeError" in r.explanation


def test_watermark_unavailable_names_the_curative_remedy(tmp_path, monkeypatch):
    """The remedy must name the whole missing stack and nothing else: the codec is vendored
    (gaige._dwtdct), so PyWavelets + OpenCV are the only installs that cure UNAVAILABLE."""

    def _no_stack():
        raise ImportError("No module named 'cv2'")

    monkeypatch.setattr(prov, "_dwtdct_codec", _no_stack)
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    r = prov.check_image_watermark(img)
    assert r.status == prov.UNAVAILABLE
    assert "PyWavelets" in r.evidence["remedy"]
    assert "opencv-python-headless" in r.evidence["remedy"]
    assert "invisible-watermark" not in r.evidence["remedy"]


# ---------------------------------------------------------------------------- C2PA checks


def test_c2pa_found_reads_the_generative_source_declaration(tmp_path):
    manifest = json.dumps({"digitalSourceType": "trainedAlgorithmicMedia"})
    r = prov.check_c2pa(tmp_path / "a.jpg", open_reader=lambda p: FakeReader(manifest))
    assert r.status == prov.FOUND
    assert r.evidence["declares_generative_source"] is True
    assert r.evidence["validation_state"] == "Valid"


def test_c2pa_no_manifest_reads_as_absent_with_the_stripping_caveat(tmp_path):
    def opener(p):
        raise RuntimeError("no JUMBF data found")

    r = prov.check_c2pa(tmp_path / "a.jpg", open_reader=opener)
    assert r.status == prov.ABSENT
    assert "stripped" in r.negative_meaning


def test_c2pa_other_failures_are_errors_not_absences(tmp_path):
    def opener(p):
        raise RuntimeError("disk on fire")

    r = prov.check_c2pa(tmp_path / "a.jpg", open_reader=opener)
    assert r.status == prov.ERROR


def test_c2pa_validation_failures_never_read_as_absent(tmp_path):
    """A claim-signature failure means a manifest IS present and something is wrong with
    it; the earlier keyword set matched "claim" and reported ABSENT for exactly this."""

    def opener(p):
        raise RuntimeError("Verify: claim signature invalid")

    r = prov.check_c2pa(tmp_path / "a.jpg", open_reader=opener)
    assert r.status == prov.ERROR


def test_c2pa_unsupported_file_type_reads_inconclusive_not_error(tmp_path):
    """Measured c2pa-python 0.37.2 behavior on non-media: NotSupported. The honest status
    is 'this carrier type cannot answer', not a crash report."""

    def opener(p):
        raise RuntimeError("NotSupported: type is unsupported")

    r = prov.check_c2pa(tmp_path / "a.doc", open_reader=opener)
    assert r.status == prov.INCONCLUSIVE
    assert "says nothing about origin" in r.negative_meaning


def test_c2pa_unavailable_names_the_remedy(tmp_path):
    def opener(p):
        raise ImportError("No module named 'c2pa'")

    r = prov.check_c2pa(tmp_path / "a.jpg", open_reader=opener)
    assert r.status == prov.UNAVAILABLE
    assert "c2pa-python" in r.evidence["remedy"]


# --------------------------------------------------------------------------------- sweeps


def test_sweep_file_refuses_a_missing_path_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="ghost.png"):
        prov.sweep_file(tmp_path / "ghost.png")


def test_sweep_text_reports_keyed_schemes_as_needs_keys():
    s = prov.sweep_text("some prose")
    assert [r["status"] for r in s["results"]] == [prov.NEEDS_KEYS, prov.NEEDS_KEYS]


def test_every_sweep_result_is_in_vocabulary_and_explains_its_negative(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    file_sweep = prov.sweep_file(
        img, codec=FakeCodec(carrier_ok=True), open_reader=lambda p: FakeReader("{}")
    )
    for sweep in (file_sweep, prov.sweep_text("x")):
        assert sweep["target"]["sha256"]
        assert sweep["note"] == prov.EVIDENCE_NOT_VERDICT
        for r in sweep["results"]:
            assert r["status"] in prov.STATUSES
            assert r["negative_meaning"]


def test_fingerprint_records_optional_libraries_including_absence():
    fp = prov.verifier_fingerprint()
    assert fp["gaige"]
    for package in ("c2pa-python", "PyWavelets", "opencv"):
        assert fp[package]  # "not installed" is itself a recorded, honest value
    # the vendored codec's row must state whose bit format it speaks
    assert "invisible-watermark 0.2.0" in fp["dwtdct-codec"]


def test_sweep_scopes_image_only_schemes_to_images(tmp_path):
    """dwtdct and synthid-image rows on a text file would imply those schemes could ever
    have applied there; only c2pa spans media types."""
    img, doc = tmp_path / "a.png", tmp_path / "a.txt"
    img.write_bytes(b"pixels")
    doc.write_text("prose", encoding="utf-8")
    opener = lambda p: FakeReader("{}")  # noqa: E731
    image_schemes = [
        r["scheme"] for r in prov.sweep_file(img, codec=FakeCodec(), open_reader=opener)["results"]
    ]
    text_schemes = [r["scheme"] for r in prov.sweep_file(doc, open_reader=opener)["results"]]
    assert image_schemes == ["c2pa", "dwtdct-image", "synthid-image"]
    assert text_schemes == ["c2pa"]


def test_render_carries_the_evidence_not_verdict_line(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"pixels")
    s = prov.sweep_file(
        img, codec=FakeCodec(carrier_ok=False), open_reader=lambda p: FakeReader("{}")
    )
    out = prov.render(s)
    assert 'No result here means "not AI"' in out
    assert "carrier self-test: FAIL" in out
    assert "a negative here means:" in out


# ------------------------------------------------------------------------------------ CLI


def test_cli_verify_text_json_round_trips(capsys):
    assert main(["verify", "--text", "hello", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {r["status"] for r in out["results"]} == {prov.NEEDS_KEYS}


def test_cli_verify_file_renders_evidence(tmp_path, capsys, monkeypatch):
    def opener(p):
        raise RuntimeError("no JUMBF data found")

    monkeypatch.setattr(prov, "_c2pa_reader", opener)
    doc = tmp_path / "note.txt"
    doc.write_text("plain file", encoding="utf-8")
    assert main(["verify", str(doc)]) == 0
    out = capsys.readouterr().out
    assert "[c2pa] ABSENT" in out
    assert "[synthid-image]" not in out  # image-only rows stay scoped to images


def test_cli_verify_missing_file_exits_with_the_remedy(tmp_path):
    with pytest.raises(SystemExit, match="ghost.png"):
        main(["verify", str(tmp_path / "ghost.png")])


def test_cli_verify_needs_a_target():
    with pytest.raises(SystemExit, match="--text"):
        main(["verify"])
