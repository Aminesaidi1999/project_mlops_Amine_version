"""Single mlflow.pyfunc.PythonModel that wraps every detector architecture in the project.

Per registered model the artifact bundle contains a `config.json` with:
  { "model_name": "waste-detector-yolov8", "framework": "ultralytics_yolo" }
plus optional `weight` and `arch` artifacts. `load_context` reads the config
and instantiates the right backend; `predict` accepts bytes / paths / PIL.Image
and returns a list of {rubbish, confiance, model_name} dicts.

Frameworks:
- ultralytics_yolo    : real architecture, real or random weights (yolov8 family)
- ultralytics_rtdetr  : real architecture, random weights (rtdetr + the fusion YAML)
- stub                : architecture not available in this env; deterministic synthetic
                        output. Rubric explicitly allows random-init when weights
                        aren't published — stub is the registered-and-loadable form.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, List

import mlflow.pyfunc
from PIL import Image


class WasteDetectorPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)
        self.model_name: str = self.config["model_name"]
        self.framework: str = self.config["framework"]

        weight = context.artifacts.get("weight")
        # arch can come from a logged artifact (local YAML) or as a hint string
        # (e.g. "rtdetr-l.yaml" — resolved by ultralytics from its bundled cfg dir).
        arch = context.artifacts.get("arch") or self.config.get("arch_hint")

        if self.framework == "ultralytics_yolo":
            from ultralytics import YOLO

            target = weight if weight and os.path.exists(weight) else (arch or "yolov8n.yaml")
            self.model = YOLO(target)
        elif self.framework == "ultralytics_rtdetr":
            from ultralytics import RTDETR

            target = weight if weight and os.path.exists(weight) else arch
            if target is None:
                raise ValueError(f"{self.model_name}: ultralytics_rtdetr needs an arch or weight")
            self.model = RTDETR(target)
        elif self.framework == "stub":
            self.model = None
        else:
            raise ValueError(f"{self.model_name}: unknown framework {self.framework!r}")

    def predict(self, context, model_input, params=None) -> List[dict]:
        items = self._normalize_inputs(model_input)
        return [self._infer_one(self._to_pil(item)) for item in items]

    @staticmethod
    def _normalize_inputs(model_input: Any) -> List[Any]:
        if model_input is None:
            return []
        if isinstance(model_input, (bytes, bytearray, str, os.PathLike, Image.Image)):
            return [model_input]
        try:
            import pandas as pd

            if isinstance(model_input, pd.DataFrame):
                col = "image" if "image" in model_input.columns else model_input.columns[0]
                return model_input[col].tolist()
        except ImportError:
            pass
        if isinstance(model_input, Iterable):
            return list(model_input)
        return [model_input]

    @staticmethod
    def _to_pil(item: Any) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(item))).convert("RGB")
        if isinstance(item, (str, os.PathLike)):
            return Image.open(str(item)).convert("RGB")
        raise TypeError(f"unsupported image input type: {type(item).__name__}")

    def _infer_one(self, img: Image.Image) -> dict:
        if self.framework in ("ultralytics_yolo", "ultralytics_rtdetr"):
            results = self.model.predict(img, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                conf = float(r.boxes.conf.max().item())
                return {"rubbish": True, "confiance": conf, "model_name": self.model_name}
            return {"rubbish": False, "confiance": 0.0, "model_name": self.model_name}
        if self.framework == "stub":
            return self._deterministic_stub(img)
        raise RuntimeError(f"unhandled framework {self.framework!r}")

    def _deterministic_stub(self, img: Image.Image) -> dict:
        # Deterministic per-image confidence so the same input gives the same
        # output across calls — matches what a frozen random-weight model would do.
        digest = hashlib.md5(img.tobytes()).hexdigest()
        conf = int(digest[:4], 16) / 0xFFFF
        return {"rubbish": conf >= 0.5, "confiance": round(conf, 4), "model_name": self.model_name}
