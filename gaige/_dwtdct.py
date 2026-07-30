# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.
#
# Derived from invisible-watermark 0.2.0 (https://github.com/ShieldMnt/invisible-watermark),
# MIT License, Copyright (c) 2021 ShieldMnt. Full notice in NOTICE.md ("Vendored code").

"""Vendored minimal dwtDct image-watermark codec: bytes payloads, encode + decode only.

Why vendored (0.0.3): invisible-watermark declares torch as a dependency and imports it at
package-import time through its rivaGan module, so its dwtDct path — which never touches
torch — is unreachable in any environment without torch installed. The install remedy gaige
shipped in 0.0.2-era nightlies ("pip install --no-deps invisible-watermark") stopped curing
UNAVAILABLE for exactly that reason. These ~100 lines are the entire dwtDct path this sweep
needs; owning them makes the codec part of the fingerprinted instrument instead of a moving
dependency, and makes the remedy (PyWavelets + OpenCV) actually curative.

Bit-format compatibility with invisible-watermark 0.2.0 is the contract: an image
watermarked by the ecosystem's encoder (Stable Diffusion pipelines use exactly this default)
must decode here byte-identically. Cross-validated in both directions against the real
library on 2026-07-30; receipts in the ops repo's 0.0.3 release prep receipt.
Three upstream oddities are therefore preserved, not fixed — changing any of them would
change the bit format:

- despite the "dct" in the scheme's name, bits are embedded directly into the DWT
  approximation band's largest-magnitude non-DC coefficient per 4x4 block (upstream
  `diffuse_dct_matrix` never applies a DCT);
- only channels 0 and 1 (Y, U of a YUV conversion) are considered, gated by per-channel
  scales whose default `(0, 36, 36)` disables Y — so the U channel alone carries the mark;
- the horizontal/vertical detail bands are swapped on the inverse DWT during embedding.
"""

from __future__ import annotations

import cv2
import numpy as np
import pywt

# Whose bit format this codec speaks (the vendoring source, pinned at the version the
# cross-validation ran against).
BITFORMAT = "invisible-watermark 0.2.0"

_SCALES = (0, 36, 36)  # per-channel quantization steps; index 0 (Y) disabled by default
_BLOCK = 4
MIN_PIXELS = 256 * 256  # upstream's contract: carriers below this cannot hold the mark


def too_small(bgr: np.ndarray) -> bool:
    """Upstream refuses carriers under 256x256 total pixels. Callers pre-check with this
    and report such carriers as unable to answer (INCONCLUSIVE), never as errors."""
    rows, cols = bgr.shape[:2]
    return rows * cols < MIN_PIXELS


def _quantize_embed(frame: np.ndarray, scale: int, bits: list[int]) -> None:
    """One bit per 4x4 block, round-robin over the payload, into the block's
    largest-magnitude non-DC coefficient (in place)."""
    rows, cols = frame.shape
    num = 0
    for i in range(rows // _BLOCK):
        for j in range(cols // _BLOCK):
            block = frame[i * _BLOCK : (i + 1) * _BLOCK, j * _BLOCK : (j + 1) * _BLOCK]
            bit = bits[num % len(bits)]
            pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
            bi, bj = pos // _BLOCK, pos % _BLOCK
            val = block[bi, bj]
            if val >= 0.0:
                block[bi, bj] = (val // scale + 0.25 + 0.5 * bit) * scale
            else:
                val = abs(val)
                block[bi, bj] = -1.0 * (val // scale + 0.25 + 0.5 * bit) * scale
            num += 1


def _quantize_infer(frame: np.ndarray, scale: int, scores: list[list[int]]) -> None:
    """The embed's inverse: vote 0/1 per block from the same coefficient's residue."""
    rows, cols = frame.shape
    num = 0
    for i in range(rows // _BLOCK):
        for j in range(cols // _BLOCK):
            block = frame[i * _BLOCK : (i + 1) * _BLOCK, j * _BLOCK : (j + 1) * _BLOCK]
            pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
            bi, bj = pos // _BLOCK, pos % _BLOCK
            val = abs(block[bi, bj])
            scores[num % len(scores)].append(int((val % scale) > 0.5 * scale))
            num += 1


def encode(bgr: np.ndarray, payload: bytes) -> np.ndarray:
    """Return a new BGR uint8 image carrying payload's bits. Raises ValueError below the
    scheme's size contract — callers decide what that means; this module never guesses."""
    if too_small(bgr):
        raise ValueError("carrier below the dwtDct 256x256-pixel minimum")
    bits = list(np.unpackbits(np.frombuffer(payload, dtype=np.uint8)))
    rows, cols = bgr.shape[:2]
    r4, c4 = rows // 4 * 4, cols // 4 * 4
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    for channel in range(2):
        if _SCALES[channel] <= 0:
            continue
        ca, (h, v, d) = pywt.dwt2(yuv[:r4, :c4, channel], "haar")
        _quantize_embed(ca, _SCALES[channel], bits)
        # upstream swaps (h, v) on the inverse transform; preserved — bit format
        yuv[:r4, :c4, channel] = pywt.idwt2((ca, (v, h, d)), "haar")
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def decode(bgr: np.ndarray, n_bits: int) -> bytes:
    """Majority-vote n_bits back out of the image; returns floor(n_bits/8) bytes.
    Raises ValueError below the size contract, same as encode."""
    if too_small(bgr):
        raise ValueError("carrier below the dwtDct 256x256-pixel minimum")
    rows, cols = bgr.shape[:2]
    r4, c4 = rows // 4 * 4, cols // 4 * 4
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    scores: list[list[int]] = [[] for _ in range(n_bits)]
    for channel in range(2):
        if _SCALES[channel] <= 0:
            continue
        ca, _details = pywt.dwt2(yuv[:r4, :c4, channel], "haar")
        _quantize_infer(ca, _SCALES[channel], scores)
    # upstream: mean vote per bit, thresholded at mean*255 > 127; empty vote lists (payload
    # longer than block capacity) read as 0, matching upstream's NaN-comparison result
    means = np.array([np.mean(s) if s else 0.0 for s in scores])
    bits = (means * 255 > 127).astype(np.uint8)
    return bytes(np.packbits(bits)[: n_bits // 8])
