from __future__ import annotations

"""Backward-compatible imports for the renamed NLI classifier module."""

from .hf_nli_clf import HFNLIClf, HFNLIClfBundle, HFNLIClfConfig

SberHFModelLoader = HFNLIClf
SberModelBundle = HFNLIClfBundle

__all__ = [
    "HFNLIClf",
    "HFNLIClfBundle",
    "HFNLIClfConfig",
    "SberHFModelLoader",
    "SberModelBundle",
]



