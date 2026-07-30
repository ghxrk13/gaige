# Test data provenance

## dwtdct_golden_384.png

sha256 `1f38c6dac0a37bd63dc9fa949dc27f9f686f9ae59a71fc49e75471c01ca750c2` · 129,957 bytes.

A 384x384 deterministic periodic-texture carrier watermarked with the payload
`StableDiffusionV1` by **real invisible-watermark 0.2.0** (`WatermarkEncoder`, method
`dwtDct`) on 2026-07-30, exactly:

```python
ii, jj = np.indices((384, 384))
base = (((ii + jj) % 8) * 24 + ((ii // 16) % 3) * 20 + 40).astype("uint8")
img = np.stack([base, np.roll(base, 3, axis=0), np.roll(base, 7, axis=1)], axis=-1)
enc = WatermarkEncoder()
enc.set_watermark("bytes", b"StableDiffusionV1")
cv2.imwrite("dwtdct_golden_384.png", enc.encode(img, "dwtDct"))
```

It pins two things at once: that the vendored codec (`gaige/_dwtdct.py`) reads what the
ecosystem's real encoder writes, and that the numeric stack (PyWavelets/OpenCV/NumPy)
still decodes it byte-identically — if this test ever reds, the instrument changed, which
is exactly the kind of event gaige exists to surface. Cross-validation receipt (all six
encode/decode directions on three carriers, plus bit-identical encoder output arrays):
the ops repo's 0.0.3 release prep receipt.

The periodic texture was chosen over the canonical seed-17 noise carrier for repository
weight only (130 KB vs 784 KB); both passed every check identically.
