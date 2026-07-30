# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Provenance evidence sweep: deterministic checks with honest negatives.

Watermarks and content credentials are the one corner of AI-text/media attribution where a
check can be deterministic: a C2PA manifest either validates or it doesn't, a watermark
payload either decodes or it doesn't. So this module emits EVIDENCE, never a score — it is
not a detector, and nothing here measures "AI-likeness".

The measurement problem hides in the negatives. A bare "no watermark found" is untrustworthy
in a carrier-dependent way: payload recoverability varies with image size, texture, and
payload length, so on an unfavourable carrier the decoder returns silence whether or not a
mark was ever embedded. The fix is a per-item instrument self-test — embed a probe payload
into *this* image and read it back. If the probe does not survive, the carrier cannot answer
the question, and the honest status is INCONCLUSIVE, not ABSENT. The same discipline covers
every other way a negative can be hollow: a scheme whose verifier needs the deployer's key
(NEEDS_KEYS), a scheme with no public verifier at all (NO_PUBLIC_DETECTOR), an optional
dependency that is not installed (UNAVAILABLE), a check that crashed (ERROR). Each result
states what a negative from it would mean, so silence can never be read as "not AI".
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__

# The closed status vocabulary. The last four exist so that "nothing found" can never be
# misread as "nothing there".
FOUND = "FOUND"
ABSENT = "ABSENT"
INCONCLUSIVE = "INCONCLUSIVE"
NEEDS_KEYS = "NEEDS_KEYS"
UNAVAILABLE = "UNAVAILABLE"
NO_PUBLIC_DETECTOR = "NO_PUBLIC_DETECTOR"
ERROR = "ERROR"
STATUSES = frozenset(
    {FOUND, ABSENT, INCONCLUSIVE, NEEDS_KEYS, UNAVAILABLE, NO_PUBLIC_DETECTOR, ERROR}
)

EVIDENCE_NOT_VERDICT = (
    "Deterministic evidence, not a verdict: each scheme reports what was checked and what a "
    'negative would mean. No result here means "not AI" — a mark can be absent, stripped, '
    "keyed, or unverifiable, and this sweep says which."
)

# Stability's default dwtDct payload for SD 1.x images, the one image watermark that is
# publicly checkable without a key. Payload bytes are compared exactly after decode.
KNOWN_IMAGE_PAYLOADS = {"StableDiffusionV1": b"StableDiffusionV1"}

# The self-test probe is the length of the longest sought payload: recoverability degrades
# with payload length, so surviving at this length certifies the carrier for every payload
# the sweep looks for.
PROBE_PAYLOAD = b"gaige-selftest-01"
assert len(PROBE_PAYLOAD) == max(len(p) for p in KNOWN_IMAGE_PAYLOADS.values())

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})


@dataclass
class SchemeResult:
    """One scheme's evidence. negative_meaning is mandatory: every negative must say what
    it does and does not establish, in the result itself rather than in documentation."""

    scheme: str
    status: str
    explanation: str
    negative_meaning: str
    evidence: dict = field(default_factory=dict)
    self_test: str | None = None  # "pass" / "fail" where a carrier self-test ran

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; statuses are {sorted(STATUSES)}")
        if self.status == ABSENT and self.self_test == "fail":
            raise ValueError(
                "ABSENT with a failed carrier self-test is the overclaim this module exists "
                "to prevent; a carrier that cannot hold the mark yields INCONCLUSIVE"
            )
        if not self.negative_meaning:
            raise ValueError("every result must state what a negative from it means")


def carrier_negative(scheme: str, probe_survived: bool) -> SchemeResult:
    """The honest-negative rule, as code: silence is ABSENT only when the carrier proved it
    could have answered."""
    if probe_survived:
        return SchemeResult(
            scheme=scheme,
            status=ABSENT,
            explanation=(
                "no known payload decoded, and a probe payload embedded into this same "
                "image survived a round trip — the carrier can hold the mark, so the "
                "silence is informative"
            ),
            negative_meaning=(
                "this image carries none of the payloads checked; it says nothing about "
                "schemes not checked or about how the image was made"
            ),
            self_test="pass",
        )
    return SchemeResult(
        scheme=scheme,
        status=INCONCLUSIVE,
        explanation=(
            "no known payload decoded, but a probe payload embedded into this same image "
            "did NOT survive a round trip — this carrier (size/texture/payload-length "
            "combination) cannot reliably hold the mark"
        ),
        negative_meaning=(
            "no information: the decoder would have returned silence here whether or not "
            "a watermark was ever embedded"
        ),
        self_test="fail",
    )


# --------------------------------------------------------------- dwtDct image watermark


def _dwtdct_codec():
    """Real codec, built lazily on the vendored gaige._dwtdct (see that module for why it
    is vendored and whose bit format it speaks). The two imports probed here are the whole
    optional imaging stack, so UNAVAILABLE stays attributable and the remedy curative."""
    import cv2  # noqa: F401 — availability probe; ImportError here means UNAVAILABLE
    import pywt  # noqa: F401

    from . import _dwtdct

    class _Codec:
        method = "dwtDct"

        def _read(self, path):
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"could not read {path} as an image")
            return img

        def decode(self, path, n_bits: int) -> bytes:
            img = self._read(path)
            if _dwtdct.too_small(img):
                # nothing this scheme could have embedded; roundtrip() below reports the
                # carrier as unable to answer, so the sweep lands on INCONCLUSIVE
                return b""
            return _dwtdct.decode(img, n_bits)

        def roundtrip(self, path, payload: bytes) -> bool:
            img = self._read(path)
            if _dwtdct.too_small(img):
                return False  # below the scheme's size contract: the carrier cannot answer
            return _dwtdct.decode(_dwtdct.encode(img, payload), len(payload) * 8) == payload

    return _Codec()


DWTDCT_REMEDY = "pip install PyWavelets opencv-python-headless"


def check_image_watermark(path: Path, codec=None) -> SchemeResult:
    """dwtDct payload check with the per-image carrier self-test. codec is injectable so the
    honesty logic is testable without the optional imaging stack."""
    scheme = "dwtdct-image"
    if codec is None:
        try:
            codec = _dwtdct_codec()
        except ImportError as e:
            return SchemeResult(
                scheme=scheme,
                status=UNAVAILABLE,
                explanation=f"imaging stack not installed ({e})",
                negative_meaning="nothing was checked; this carries no information",
                evidence={"remedy": DWTDCT_REMEDY},
            )
    try:
        for name, payload in KNOWN_IMAGE_PAYLOADS.items():
            if codec.decode(path, len(payload) * 8) == payload:
                return SchemeResult(
                    scheme=scheme,
                    status=FOUND,
                    explanation=f"recovered the {name} payload via {codec.method}",
                    negative_meaning="n/a — payload recovered",
                    evidence={"payload": name, "method": codec.method},
                )
        return carrier_negative(scheme, codec.roundtrip(path, PROBE_PAYLOAD))
    except Exception as e:  # a crashed check must degrade to a loud status, never a verdict
        return SchemeResult(
            scheme=scheme,
            status=ERROR,
            explanation=f"{type(e).__name__}: {e}",
            negative_meaning="the check crashed; this carries no information",
        )


# ------------------------------------------------------------------ C2PA content credentials


def _c2pa_reader(path: Path):
    from c2pa import Reader

    return Reader(str(path))


def _classify_reader_error(e: Exception) -> str:
    """Absence must be a typed signal, never a keyword guess. Measured against c2pa-python
    0.37.2 (0.0.3 release prep receipts): a manifest-less file raises ManifestNotFound
    ("no JUMBF data found"), an unsupported file type raises NotSupported, and a corrupted
    or invalid manifest raises Verify-class errors. Only the first may read ABSENT — the
    earlier substring set here included "claim", which routed claim-signature validation
    failures to ABSENT, the exact overclaim this module exists to prevent. Class name and
    message are matched together so injectable readers stay testable."""
    ident = f"{type(e).__name__} {e}".lower()
    if "manifestnotfound" in ident or "no jumbf" in ident:
        return ABSENT
    if "notsupported" in ident or "type is unsupported" in ident:
        return INCONCLUSIVE
    return ERROR


def check_c2pa(path: Path, open_reader=None) -> SchemeResult:
    """C2PA Content Credentials: the one cross-modality provenance signal that is publicly
    checkable, and the checkable side of vendor watermarks that ship credentials alongside."""
    scheme = "c2pa"
    negative = (
        "credentials are routinely stripped in ordinary handling (re-encode, resize, "
        "screenshot, most upload pipelines); absence says nothing about origin"
    )
    opener = open_reader or _c2pa_reader
    try:
        reader = opener(path)
    except ImportError as e:
        return SchemeResult(
            scheme=scheme,
            status=UNAVAILABLE,
            explanation=f"c2pa sdk not installed ({e})",
            negative_meaning="nothing was checked; this carries no information",
            evidence={"remedy": "pip install c2pa-python"},
        )
    except Exception as e:
        kind = _classify_reader_error(e)
        if kind == ABSENT:
            return SchemeResult(
                scheme=scheme,
                status=ABSENT,
                explanation="no C2PA manifest in the file",
                negative_meaning=negative,
            )
        if kind == INCONCLUSIVE:
            return SchemeResult(
                scheme=scheme,
                status=INCONCLUSIVE,
                explanation=f"the C2PA SDK does not support this file type ({e})",
                negative_meaning=(
                    "this file type could not be checked for credentials; "
                    "it says nothing about origin"
                ),
            )
        return SchemeResult(
            scheme=scheme,
            status=ERROR,
            explanation=f"{type(e).__name__}: {e}",
            negative_meaning="the check crashed; this carries no information",
        )
    try:
        manifest = reader.json()
        state = getattr(reader, "get_validation_state", lambda: "unknown")()
        return SchemeResult(
            scheme=scheme,
            status=FOUND,
            explanation="C2PA manifest present; validation state and any declared "
            "generative-AI source type are in the evidence",
            negative_meaning="n/a — manifest recovered",
            evidence={
                "validation_state": str(state),
                # the standardized machine-readable "made by generative AI" declaration
                "declares_generative_source": "trainedAlgorithmicMedia" in manifest,
            },
        )
    except Exception as e:
        return SchemeResult(
            scheme=scheme,
            status=ERROR,
            explanation=f"{type(e).__name__}: {e}",
            negative_meaning="the manifest could not be parsed; this carries no information",
        )


# ------------------------------------------------------------------------- keyed / portal


def check_text_watermarks() -> list[SchemeResult]:
    """Text watermarks are keyed: without the deployer's watermarking key there is nothing
    to measure, and any tool claiming otherwise is checking something else."""
    keyed = "the check cannot run without the deployer's key; this carries no information"
    return [
        SchemeResult(
            scheme="synthid-text",
            status=NEEDS_KEYS,
            explanation="keyed scheme; verification requires the deployer's watermarking "
            "config and detector weights",
            negative_meaning=keyed,
        ),
        SchemeResult(
            scheme="kgw-text",
            status=NEEDS_KEYS,
            explanation="keyed red/green-list scheme; verification requires the deployer's "
            "hash key and generation settings",
            negative_meaning=keyed,
        ),
    ]


def _portal_only() -> SchemeResult:
    return SchemeResult(
        scheme="synthid-image",
        status=NO_PUBLIC_DETECTOR,
        explanation="verifiable only inside the vendor's own portal; no public verifier "
        "exists. The locally checkable signal for such images is the C2PA manifest "
        "shipped alongside, checked above",
        negative_meaning="nothing local can check this scheme; this carries no information",
    )


# --------------------------------------------------------------------------------- sweeps


def verifier_fingerprint() -> dict:
    """What ran the sweep. Optional libraries are recorded honestly, including their absence
    — an UNAVAILABLE status must be attributable to the environment that produced it."""
    import importlib

    fp = {"gaige": __version__, "python": platform.python_version()}
    for module, package in (
        ("c2pa", "c2pa-python"),
        ("pywt", "PyWavelets"),
        ("cv2", "opencv"),
    ):
        try:
            m = importlib.import_module(module)
            fp[package] = str(getattr(m, "__version__", "installed, version unrecorded"))
        except ImportError:
            fp[package] = "not installed"
    # The codec is in-tree (gaige._dwtdct), so its version rides gaige's own; this row
    # records whose bit format it speaks. Kept a literal so the fingerprint never needs
    # the optional imaging stack the codec imports.
    fp["dwtdct-codec"] = "vendored; bit-format invisible-watermark 0.2.0"
    return fp


def sweep_file(path: str | Path, codec=None, open_reader=None) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"no such file: {p} — gaige verify sweeps an existing local file (or use --text)"
        )
    results = [check_c2pa(p, open_reader=open_reader)]
    if p.suffix.lower() in IMAGE_SUFFIXES:
        results.append(check_image_watermark(p, codec=codec))
        # image-only scheme rows stay scoped to images: a NO_PUBLIC_DETECTOR row on a
        # text file would imply the scheme could ever have applied there
        results.append(_portal_only())
    return {
        "target": {
            "kind": "file",
            "name": p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        },
        "note": EVIDENCE_NOT_VERDICT,
        "fingerprint": verifier_fingerprint(),
        "results": [asdict(r) for r in results],
    }


def sweep_text(text: str) -> dict:
    return {
        "target": {
            "kind": "text",
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        "note": EVIDENCE_NOT_VERDICT,
        "fingerprint": verifier_fingerprint(),
        "results": [asdict(r) for r in check_text_watermarks()],
    }


def render(sweep: dict) -> str:
    t = sweep["target"]
    ident = t["name"] if t["kind"] == "file" else f"{t['chars']} chars"
    fp = " · ".join(f"{k} {v}" for k, v in sweep["fingerprint"].items())
    lines = [
        "gaige verify — provenance evidence sweep",
        f"target: {t['kind']} {ident} · sha256 {t['sha256'][:16]}...",
        f"instrument: {fp}",
        "",
        sweep["note"],
        "",
    ]
    for r in sweep["results"]:
        lines.append(f"[{r['scheme']}] {r['status']} — {r['explanation']}")
        if r["self_test"]:
            lines.append(
                f"    carrier self-test: {r['self_test'].upper()} "
                "(probe payload embedded into this image and read back)"
            )
        if r["status"] != FOUND:
            lines.append(f"    a negative here means: {r['negative_meaning']}")
        for k, v in r["evidence"].items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines) + "\n"
