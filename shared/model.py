"""Dependency-free scoring for the JSON logistic-regression artifact.

The artifact is plain JSON (not pickle) so loading it from S3 can never execute
arbitrary code, and the API needs no scikit-learn at runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

from shared.features import MODEL_FEATURES


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    version: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict) -> "LogisticModel":
        names = tuple(data["feature_names"])
        if names != MODEL_FEATURES:
            raise ValueError(
                f"model features {names} do not match serving features {MODEL_FEATURES}"
            )
        model = cls(
            feature_names=names,
            coefficients=tuple(float(c) for c in data["coefficients"]),
            intercept=float(data["intercept"]),
            means=tuple(float(m) for m in data["means"]),
            scales=tuple(float(s) or 1.0 for s in data["scales"]),
            version=str(data.get("version", "unknown")),
        )
        if not (len(model.coefficients) == len(model.means) == len(model.scales) == len(names)):
            raise ValueError("model artifact vectors have inconsistent lengths")
        return model

    @classmethod
    def from_json(cls, raw: str | bytes) -> "LogisticModel":
        return cls.from_dict(json.loads(raw))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "means": list(self.means),
            "scales": list(self.scales),
        }

    def predict_proba(self, vector: Sequence[float]) -> float:
        z = self.intercept
        for x, coef, mean, scale in zip(vector, self.coefficients, self.means, self.scales):
            z += coef * (x - mean) / scale
        # Numerically stable sigmoid.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)
