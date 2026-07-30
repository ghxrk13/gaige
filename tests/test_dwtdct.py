# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The vendored dwtDct codec against real images (the logic-level honesty rules are covered
in test_provenance.py through injectable fakes). Everything here needs the optional imaging
stack, so the whole module skips on the numpy+requests core — the suite's core-only promise
holds either way.

The golden test is the load-bearing one: the banked PNG was watermarked by real
invisible-watermark 0.2.0 (provenance + regeneration recipe in tests/data/README.md), so it
pins ecosystem bit-compatibility AND numeric-stack stability in a single byte comparison. A
red here means the instrument changed, not (necessarily) the code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pywt")

from gaige import _dwtdct  # noqa: E402
from gaige import provenance as prov  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "dwtdct_golden_384.png"
SD_PAYLOAD = b"StableDiffusionV1"


def textured_carrier(n=512, seed=17):
    """The canonical fixture recipe: seeded noise plus text, texture enough that a 136-bit
    payload survives (measured 2026-07-21: it does not at 256x256 or on low texture)."""
    rng = np.random.default_rng(seed)
    img = (rng.random((n, n, 3)) * 255).astype("uint8")
    cv2.putText(img, "plain", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    return img


def test_golden_from_real_invisible_watermark_decodes_byte_identically():
    marked = cv2.imread(str(GOLDEN))
    assert marked is not None, "golden fixture missing or unreadable"
    assert _dwtdct.decode(marked, len(SD_PAYLOAD) * 8) == SD_PAYLOAD


def test_roundtrip_on_the_canonical_textured_carrier():
    img = textured_carrier()
    marked = _dwtdct.encode(img, prov.PROBE_PAYLOAD)
    assert _dwtdct.decode(marked, len(prov.PROBE_PAYLOAD) * 8) == prov.PROBE_PAYLOAD


def test_encode_is_deterministic_and_leaves_its_input_alone():
    img = textured_carrier()
    before = img.copy()
    m1 = _dwtdct.encode(img, SD_PAYLOAD)
    m2 = _dwtdct.encode(img, SD_PAYLOAD)
    assert np.array_equal(m1, m2)
    assert np.array_equal(img, before)


def test_payload_survives_the_png_write_read_path(tmp_path):
    """The sweep reads files, not arrays; the mark must survive lossless serialization."""
    p = tmp_path / "marked.png"
    cv2.imwrite(str(p), _dwtdct.encode(textured_carrier(), SD_PAYLOAD))
    r = prov.check_image_watermark(p)
    assert r.status == prov.FOUND
    assert r.evidence["payload"] == "StableDiffusionV1"


def test_a_carrier_below_the_size_contract_reads_inconclusive_not_error(tmp_path):
    """Upstream refuses sub-256x256-pixel carriers; the honest sweep answer is 'this
    carrier cannot answer', never a crash and never ABSENT."""
    with pytest.raises(ValueError, match="256x256"):
        _dwtdct.encode(np.full((100, 100, 3), 128, dtype="uint8"), b"x")
    small = tmp_path / "small.png"
    cv2.imwrite(str(small), np.full((200, 200, 3), 200, dtype="uint8"))
    r = prov.check_image_watermark(small)
    assert (r.status, r.self_test) == (prov.INCONCLUSIVE, "fail")
