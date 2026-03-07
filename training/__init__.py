"""Training utilities for CognitiveTwin."""
from .losses import CognitiveTwinLoss
from .train_multimodal import main as train_main

__all__ = ["CognitiveTwinLoss", "train_main"]
