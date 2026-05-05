"""
DDColor (Tencent, 2023) wrapper for B&W -> color on historical photos.

Loads via the official modelscope pipeline. The first call downloads ~600 MB
of weights from ModelScope's CDN to ~/.cache/modelscope.

Standalone usage:
    python scripts/colorize.py --input photo_bw.jpg --output photo_color.png

Pipeline integration:
    from colorizer import Colorizer
    c = Colorizer()
    img_color = c.colorize(img_bw_bgr)
"""

from __future__ import annotations

import numpy as np


DEFAULT_MODEL = "damo/cv_ddcolor_image-colorization"


class Colorizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "gpu"):
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as e:
            raise ImportError(
                "Colorizer requires modelscope. Run:\n"
                "    .\\setup.ps1 -Colorize\n"
                "or:\n"
                "    pip install modelscope"
            ) from e

        from modelscope.outputs import OutputKeys
        self._OutputKeys = OutputKeys
        print(f"[colorizer] loading {model_name} (first call downloads ~600 MB)")
        self.pipe = pipeline(Tasks.image_colorization, model=model_name, device=device)
        print("[colorizer] ready")

    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        """Colorize a BGR uint8 image. Returns BGR uint8."""
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got {img_bgr.shape}")
        result = self.pipe(img_bgr)
        out = result[self._OutputKeys.OUTPUT_IMG]
        if out is None:
            raise RuntimeError("DDColor returned no image")
        return out
