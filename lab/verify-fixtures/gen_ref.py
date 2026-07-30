"""Live-arm fixtures, reference side (runs in refenv with REAL invisible-watermark).

The watermark positive is encoded by the real ecosystem library so the live arm proves the
vendored codec reads real marks from disk, end to end. Also emits the negative control and
the below-size-contract carrier."""

from pathlib import Path

import cv2
import numpy as np
from imwatermark import WatermarkEncoder

FIX = Path(__file__).parent / "fixtures"
FIX.mkdir(exist_ok=True)

# negative control: 512x512 textured (measured 2026-07-21: a 136-bit payload does not
# survive at 256x256 or on low texture, so a smaller/smoother fixture would test nothing)
rng = np.random.default_rng(17)
img = (rng.random((512, 512, 3)) * 255).astype("uint8")
cv2.putText(img, "plain", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
cv2.imwrite(str(FIX / "plain.png"), img)
print("[fixture] plain.png")

# positive control: the same carrier with the Stable Diffusion 1.x payload, REAL encoder
enc = WatermarkEncoder()
enc.set_watermark("bytes", b"StableDiffusionV1")
cv2.imwrite(str(FIX / "sd_watermarked.png"), enc.encode(img.copy(), "dwtDct"))
print("[fixture] sd_watermarked.png (real invisible-watermark 0.2.0 encoder)")

# below the scheme's 256x256-pixel contract: the carrier that cannot answer
cv2.imwrite(str(FIX / "small200.png"), np.full((200, 200, 3), 200, dtype="uint8"))
print("[fixture] small200.png")

# jpg source for the c2pa arm (signed in the bench env, which has c2pa-python)
cv2.imwrite(str(FIX / "plain.jpg"), img)
print("[fixture] plain.jpg (c2pa signing source)")
