import csv
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .format import remove_underline
from .overlap import drop_overlap_tags


_LABEL_CACHE: dict[tuple[str, bool], tuple[list[str], dict[int, list[int]], dict[str, list[str]]]] = {}
_THRESHOLD_CACHE: dict[str, tuple[dict[int, float], dict[int, str]]] = {}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_pixai_labels(csv_path: Path, no_underline: bool = False) -> tuple[list[str], dict[int, list[int]], dict[str, list[str]]]:
    key = (str(csv_path), no_underline)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]

    tag_names = []
    indexes_by_category: dict[int, list[int]] = {}
    ips_mapping: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        if records and "id" in records[0]:
            records.sort(key=lambda item: int(item.get("id") or 0))

        for index, row in enumerate(records):
            tag = row.get("name", "")
            if no_underline:
                tag = remove_underline(tag)
            tag_names.append(tag)

            if "category" in row and row.get("category") != "":
                category = int(row.get("category", 0) or 0)
            else:
                category = 4 if str(row.get("is_char", "")).lower() in {"1", "true", "yes"} else 0
            indexes_by_category.setdefault(category, []).append(index)

            ips = row.get("ips", "")
            if ips:
                try:
                    values = json.loads(ips)
                except json.JSONDecodeError:
                    try:
                        values = ast.literal_eval(ips)
                    except Exception:
                        values = []
                if values:
                    ips_mapping[tag] = values

    value = (tag_names, indexes_by_category, ips_mapping)
    _LABEL_CACHE[key] = value
    return value


def load_pixai_thresholds(path: Path | None) -> tuple[dict[int, float], dict[int, str]]:
    if path is None or not path.exists():
        return {0: 0.30, 4: 0.85}, {0: "general", 4: "character"}

    key = str(path)
    if key in _THRESHOLD_CACHE:
        return _THRESHOLD_CACHE[key]

    thresholds: dict[int, float] = {}
    category_names: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = int(row["category"])
            thresholds[category] = float(row["threshold"])
            category_names[category] = row.get("name", str(category))

    thresholds.setdefault(0, 0.30)
    thresholds.setdefault(4, 0.85)
    category_names.setdefault(0, "general")
    category_names.setdefault(4, "character")
    value = (thresholds, category_names)
    _THRESHOLD_CACHE[key] = value
    return value


def _target_size_from_session(session) -> int:
    shape = session.get_inputs()[0].shape
    int_dims = [dim for dim in shape if isinstance(dim, int)]
    if len(int_dims) >= 2:
        return int(max(int_dims[-2:]))
    return 448


def _normalize_from_preprocess(preprocess_path: Path | None) -> tuple[list[float], list[float]]:
    if preprocess_path is None or not preprocess_path.exists():
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    try:
        data = json.loads(preprocess_path.read_text(encoding="utf-8"))
    except Exception:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    text = json.dumps(data).lower()
    if "imagenet" in text:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def prepare_image_for_pixai(image_np: np.ndarray, target_size: int, preprocess_path: Path | None = None) -> np.ndarray:
    image = Image.fromarray(image_np, mode="RGB")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)

    array = np.asarray(image, dtype=np.float32) / 255.0
    mean, std = _normalize_from_preprocess(preprocess_path)
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    array = array.transpose(2, 0, 1)
    return np.expand_dims(array, axis=0).astype(np.float32)


def postprocess_pixai(
    pred: np.ndarray,
    csv_path: Path,
    thresholds_path: Path | None = None,
    threshold: float | None = None,
    character_threshold: float | None = None,
    no_underline: bool = False,
    drop_overlap: bool = False,
) -> dict[str, dict[str, float] | list[str] | np.ndarray]:
    tag_names, indexes_by_category, ips_mapping_all = load_pixai_labels(csv_path, no_underline)
    default_thresholds, category_names = load_pixai_thresholds(thresholds_path)
    if threshold is not None:
        default_thresholds[0] = threshold
    if character_threshold is not None:
        default_thresholds[4] = character_threshold

    result: dict[str, dict[str, float] | list[str] | np.ndarray] = {}
    all_tags: dict[str, float] = {}
    for category, indexes in indexes_by_category.items():
        category_name = category_names.get(category, str(category))
        category_threshold = default_thresholds.get(category, threshold if threshold is not None else 0.35)
        selected = {
            tag_names[index]: float(pred[index])
            for index in indexes
            if index < len(pred) and pred[index] > category_threshold
        }
        if drop_overlap and category == 0:
            selected = drop_overlap_tags(selected)
        result[category_name] = selected
        if category != 9:
            all_tags.update(selected)

    character_tags = result.get("character", {})
    ips_mapping = {
        tag: ips_mapping_all[tag]
        for tag in character_tags
        if tag in ips_mapping_all
    } if isinstance(character_tags, dict) else {}
    ips_count = Counter(ip for values in ips_mapping.values() for ip in values)

    result["tag"] = all_tags
    result["ips_mapping"] = ips_mapping
    result["ips_count"] = dict(ips_count)
    result["ips"] = [ip for ip, _ in sorted(ips_count.items(), key=lambda item: (-item[1], item[0]))]
    result["prediction"] = pred.astype(np.float32)
    return result


def get_pixai_tags(
    image_np: np.ndarray,
    session,
    csv_path: Path,
    thresholds_path: Path | None = None,
    preprocess_path: Path | None = None,
    threshold: float | None = None,
    character_threshold: float | None = None,
    no_underline: bool = False,
    drop_overlap: bool = False,
) -> dict[str, dict[str, float] | list[str] | np.ndarray]:
    input_info = session.get_inputs()[0]
    input_np = prepare_image_for_pixai(image_np, _target_size_from_session(session), preprocess_path)
    output_names = [output.name for output in session.get_outputs()]
    output_values = session.run(output_names, {input_info.name: input_np})
    output_map = {name: value[0] for name, value in zip(output_names, output_values)}
    pred = output_map.get("prediction")
    if pred is None:
        pred = output_map.get("logits")
        if pred is None:
            pred = output_values[0][0]
        pred = sigmoid(pred)

    return postprocess_pixai(
        pred,
        csv_path=csv_path,
        thresholds_path=thresholds_path,
        threshold=threshold,
        character_threshold=character_threshold,
        no_underline=no_underline,
        drop_overlap=drop_overlap,
    )
