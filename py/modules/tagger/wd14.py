import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .format import remove_underline
from .overlap import drop_overlap_tags


_LABEL_CACHE: dict[tuple[str, bool], tuple[list[str], list[int], list[int], list[int]]] = {}


def load_wd14_labels(csv_path: Path, no_underline: bool = False) -> tuple[list[str], list[int], list[int], list[int]]:
    key = (str(csv_path), no_underline)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]

    tag_names = []
    rating_indexes = []
    general_indexes = []
    character_indexes = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            tag = row["name"]
            if no_underline:
                tag = remove_underline(tag)
            tag_names.append(tag)

            category = int(row.get("category", -1))
            if category == 9:
                rating_indexes.append(index)
            elif category == 0:
                general_indexes.append(index)
            elif category == 4:
                character_indexes.append(index)

    value = (tag_names, rating_indexes, general_indexes, character_indexes)
    _LABEL_CACHE[key] = value
    return value


def mcut_threshold(probs: np.ndarray) -> float:
    if len(probs) < 2:
        return 0.0
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    index = int(difs.argmax())
    return float((sorted_probs[index] + sorted_probs[index + 1]) / 2)


def prepare_image_for_wd14(image_np: np.ndarray, target_size: int) -> np.ndarray:
    image = Image.fromarray(image_np, mode="RGB")
    max_dim = max(image.size)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))
    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.Resampling.BICUBIC)
    array = np.asarray(padded, dtype=np.float32)
    array = array[:, :, ::-1]
    return np.expand_dims(array, axis=0)


def postprocess_wd14(
    pred: np.ndarray,
    csv_path: Path,
    threshold: float = 0.35,
    character_threshold: float = 0.85,
    no_underline: bool = False,
    general_mcut_enabled: bool = False,
    character_mcut_enabled: bool = False,
    drop_overlap: bool = False,
) -> dict[str, dict[str, float] | np.ndarray]:
    tag_names, rating_indexes, general_indexes, character_indexes = load_wd14_labels(csv_path, no_underline)
    labels = list(zip(tag_names, pred.astype(float)))

    rating = {labels[i][0]: float(labels[i][1]) for i in rating_indexes}
    general_candidates = [labels[i] for i in general_indexes]
    character_candidates = [labels[i] for i in character_indexes]

    if general_mcut_enabled:
        threshold = mcut_threshold(np.array([score for _, score in general_candidates]))
    if character_mcut_enabled:
        character_threshold = max(0.15, mcut_threshold(np.array([score for _, score in character_candidates])))

    general = {tag: float(score) for tag, score in general_candidates if score > threshold}
    if drop_overlap:
        general = drop_overlap_tags(general)
    character = {tag: float(score) for tag, score in character_candidates if score > character_threshold}

    return {
        "rating": rating,
        "general": general,
        "character": character,
        "tag": {**character, **general},
        "prediction": pred.astype(np.float32),
    }


def get_wd14_tags(
    image_np: np.ndarray,
    session,
    csv_path: Path,
    threshold: float = 0.35,
    character_threshold: float = 0.85,
    no_underline: bool = False,
    drop_overlap: bool = False,
) -> dict[str, dict[str, float] | np.ndarray]:
    input_info = session.get_inputs()[0]
    target_size = int(input_info.shape[1])
    input_np = prepare_image_for_wd14(image_np, target_size)
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_info.name: input_np})[0][0]
    return postprocess_wd14(
        pred,
        csv_path=csv_path,
        threshold=threshold,
        character_threshold=character_threshold,
        no_underline=no_underline,
        drop_overlap=drop_overlap,
    )
