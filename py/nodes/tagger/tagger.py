import csv
from pathlib import Path

from comfy_api.latest import io
from .common import PACKAGE_NAME, CATEGORY
from ...utils import mk_name

import comfy.utils
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from . import model_manager


class Tagger(io.ComfyNode):
    _sessions: dict[str, ort.InferenceSession] = {}
    _tag_cache: dict[str, tuple[list[str], int, int]] = {}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "Tagger"),
            display_name="Tagger",
            category=CATEGORY,
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("model", options=model_manager.KNOWN_TAGGERS),
                io.Float.Input("threshold", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("character_threshold", default=0.85, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("replace_underscore", default=True),
                io.Boolean.Input("trailing_comma", default=False),
                io.String.Input("exclude_tags", default=""),
            ],
            outputs=[
                io.String.Output(is_output_list=True),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        model: str,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        exclude_tags: str = "",
    ):
        manager = model_manager.ModelManager()
        download_pbar = comfy.utils.ProgressBar(1)
        if not manager.download_model(model, download_pbar):
            raise RuntimeError(f"Failed to prepare tagger model: {model}")
        download_pbar.update_absolute(1, 1)

        session = cls._get_session(manager.get_model_path(model, "onnx"))
        tags, general_index, character_index = cls._load_tags(
            manager.get_model_path(model, "csv"),
            replace_underscore,
        )

        images_np = cls._images_to_uint8(images)
        input_info = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
        input_size = int(input_info.shape[1])
        exclude = {tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()}

        results = []
        pbar = comfy.utils.ProgressBar(images_np.shape[0])
        for image_np in images_np:
            input_np = cls._preprocess_image(image_np, input_size)
            probs = session.run([output_name], {input_info.name: input_np})[0][0]
            result = cls._format_tags(
                tags,
                probs,
                general_index,
                character_index,
                threshold,
                character_threshold,
                exclude,
                trailing_comma,
            )
            results.append(result)
            pbar.update(1)

        return io.NodeOutput(results)

    @classmethod
    def _get_session(cls, model_path: Path):
        key = str(model_path)
        if key in cls._sessions:
            return cls._sessions[key]

        providers = [
            provider
            for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider in ort.get_available_providers()
        ]
        if not providers:
            providers = ort.get_available_providers()

        session = ort.InferenceSession(key, providers=providers)
        cls._sessions[key] = session
        return session

    @classmethod
    def _load_tags(cls, csv_path: Path, replace_underscore: bool):
        key = f"{csv_path}:{replace_underscore}"
        if key in cls._tag_cache:
            return cls._tag_cache[key]

        tags = []
        general_index = None
        character_index = None
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2

                tag = row[1]
                if replace_underscore:
                    tag = tag.replace("_", " ")
                tags.append(tag)

        if general_index is None or character_index is None:
            raise ValueError(f"Could not find tag categories in {csv_path}")

        value = (tags, general_index, character_index)
        cls._tag_cache[key] = value
        return value

    @staticmethod
    def _images_to_uint8(images: torch.Tensor):
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

    @staticmethod
    def _preprocess_image(image_np: np.ndarray, input_size: int):
        image = Image.fromarray(image_np, mode="RGB")
        ratio = float(input_size) / max(image.size)
        new_size = tuple(max(1, int(x * ratio)) for x in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        square = Image.new("RGB", (input_size, input_size), (255, 255, 255))
        square.paste(image, ((input_size - new_size[0]) // 2, (input_size - new_size[1]) // 2))

        array = np.asarray(square).astype(np.float32)
        array = array[:, :, ::-1]
        return np.expand_dims(array, 0)

    @classmethod
    def _format_tags(
        cls,
        tags: list[str],
        probs: np.ndarray,
        general_index: int,
        character_index: int,
        threshold: float,
        character_threshold: float,
        exclude: set[str],
        trailing_comma: bool,
    ):
        result = list(zip(tags, probs))
        general = [item for item in result[general_index:character_index] if item[1] > threshold]
        character = [item for item in result[character_index:] if item[1] > character_threshold]

        selected = character + general
        selected = [item for item in selected if item[0].lower() not in exclude]

        if trailing_comma:
            return "".join(cls._escape_tag(item[0]) + ", " for item in selected)
        return ", ".join(cls._escape_tag(item[0]) for item in selected)

    @staticmethod
    def _escape_tag(tag: str):
        return tag.replace("(", "\\(").replace(")", "\\)")
