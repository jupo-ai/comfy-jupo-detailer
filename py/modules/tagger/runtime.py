from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


_SESSIONS: dict[str, ort.InferenceSession] = {}


def open_onnx_session(model_path: Path) -> ort.InferenceSession:
    key = str(model_path)
    if key in _SESSIONS:
        return _SESSIONS[key]

    providers = [
        provider
        for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider in ort.get_available_providers()
    ]
    if not providers:
        providers = ort.get_available_providers()

    session = ort.InferenceSession(key, providers=providers)
    _SESSIONS[key] = session
    return session


def images_to_uint8(images: torch.Tensor) -> np.ndarray:
    if images.dim() != 4:
        raise ValueError(f"Expected images [B, H, W, C], got {images.dim()}D (shape: {images.shape})")

    images = images.detach().cpu().clamp(0.0, 1.0)
    if images.shape[-1] == 1:
        images = images.expand(*images.shape[:-1], 3)
    elif images.shape[-1] > 3:
        images = images[..., :3]
    elif images.shape[-1] != 3:
        raise ValueError(f"Expected image channels 1, 3, or 4, got shape: {images.shape}")

    return (images.numpy() * 255.0).round().astype(np.uint8)
