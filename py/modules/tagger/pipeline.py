from pathlib import Path
from typing import Protocol

import torch

from .blacklist import drop_blacklisted_tags
from .character import drop_basic_character_tags
from .format import tags_to_text
from .order import SortMode, sort_tags
from .registry import get_model_spec
from .runtime import images_to_uint8, open_onnx_session
from .wd14 import get_wd14_tags
from .pixai import get_pixai_tags


class ProgressBarLike(Protocol):
    def update(self, value: int) -> None:
        ...


def _update_progress(pbar: ProgressBarLike | None, value: int = 1) -> None:
    if pbar is not None:
        pbar.update(value)


def _exclude_tags(tags: dict[str, float], exclude_tags: str) -> dict[str, float]:
    exclude = {tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()}
    if not exclude:
        return tags
    return {tag: score for tag, score in tags.items() if tag.lower() not in exclude}


def _filter_tags(
    tags: dict[str, float],
    exclude_tags: str,
    drop_blacklist: bool,
    drop_basic_character: bool,
) -> dict[str, float]:
    tags = _exclude_tags(tags, exclude_tags)
    if drop_blacklist:
        tags = drop_blacklisted_tags(tags)
    if drop_basic_character:
        tags = drop_basic_character_tags(tags)
    return tags


def _sort_tag_mapping(
    tags: dict[str, float],
    sort_mode: SortMode,
    prioritize_people_tags: bool,
) -> dict[str, float]:
    return {tag: tags[tag] for tag in sort_tags(tags, mode=sort_mode, prioritize_people_tags=prioritize_people_tags)}


def _ordered_tags_from_output(
    output: dict,
    exclude_tags: str,
    drop_blacklist: bool,
    drop_basic_character: bool,
    sort_mode: SortMode,
    prioritize_people_tags: bool,
) -> dict[str, float]:
    grouped = []
    for group_name in ("character", "general"):
        group_tags = output.get(group_name)
        if isinstance(group_tags, dict):
            group_tags = _filter_tags(group_tags, exclude_tags, drop_blacklist, drop_basic_character)
            grouped.append(_sort_tag_mapping(group_tags, sort_mode, prioritize_people_tags))

    if grouped:
        ordered_tags = {}
        for group_tags in grouped:
            ordered_tags.update(group_tags)

        all_tags = output.get("tag", {})
        if isinstance(all_tags, dict):
            remaining = {tag: score for tag, score in all_tags.items() if tag not in ordered_tags}
            remaining = _filter_tags(remaining, exclude_tags, drop_blacklist, drop_basic_character)
            ordered_tags.update(_sort_tag_mapping(remaining, sort_mode, prioritize_people_tags))
        return ordered_tags

    tags = output.get("tag", {})
    if not isinstance(tags, dict):
        return {}
    tags = _filter_tags(tags, exclude_tags, drop_blacklist, drop_basic_character)
    return _sort_tag_mapping(tags, sort_mode, prioritize_people_tags)


def _grouped_tags_from_output(
    output: dict,
    exclude_tags: str,
    drop_blacklist: bool,
    drop_basic_character: bool,
    sort_mode: SortMode,
    prioritize_people_tags: bool,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    used_tags = set()

    for group_name in ("character", "general"):
        group_tags = output.get(group_name)
        if isinstance(group_tags, dict):
            group_tags = _filter_tags(group_tags, exclude_tags, drop_blacklist, drop_basic_character)
            group_tags = _sort_tag_mapping(group_tags, sort_mode, prioritize_people_tags)
            grouped[group_name] = group_tags
            used_tags.update(group_tags)
        else:
            grouped[group_name] = {}

    all_tags = output.get("tag", {})
    if isinstance(all_tags, dict):
        other_tags = {tag: score for tag, score in all_tags.items() if tag not in used_tags}
        other_tags = _filter_tags(other_tags, exclude_tags, drop_blacklist, drop_basic_character)
        grouped["other"] = _sort_tag_mapping(other_tags, sort_mode, prioritize_people_tags)
    else:
        grouped["other"] = {}

    return grouped


def tag_images_grouped(
    images: torch.Tensor,
    model: str,
    model_paths: dict[str, Path],
    threshold: float,
    character_threshold: float,
    replace_underscore: bool = True,
    exclude_tags: str = "",
    drop_overlap: bool = False,
    drop_blacklist: bool = False,
    drop_basic_character: bool = False,
    sort_mode: SortMode = "original",
    prioritize_people_tags: bool = True,
    pbar: ProgressBarLike | None = None,
) -> list[dict[str, dict[str, float]]]:
    spec = get_model_spec(model)
    session = open_onnx_session(model_paths["model.onnx"])
    images_np = images_to_uint8(images)

    results = []
    for image_np in images_np:
        if spec.family == "wd14":
            output = get_wd14_tags(
                image_np,
                session,
                csv_path=model_paths["selected_tags.csv"],
                threshold=threshold,
                character_threshold=character_threshold,
                no_underline=replace_underscore,
                drop_overlap=drop_overlap,
            )
        elif spec.family == "pixai":
            output = get_pixai_tags(
                image_np,
                session,
                csv_path=model_paths["selected_tags.csv"],
                thresholds_path=model_paths.get("thresholds.csv"),
                preprocess_path=model_paths.get("preprocess.json"),
                threshold=threshold,
                character_threshold=character_threshold,
                no_underline=replace_underscore,
                drop_overlap=drop_overlap,
            )
        else:
            raise ValueError(f"Unsupported tagger family: {spec.family}")

        results.append(
            _grouped_tags_from_output(
                output,
                exclude_tags=exclude_tags,
                drop_blacklist=drop_blacklist,
                drop_basic_character=drop_basic_character,
                sort_mode=sort_mode,
                prioritize_people_tags=prioritize_people_tags,
            )
        )
        _update_progress(pbar)

    return results


def tag_images(
    images: torch.Tensor,
    model: str,
    model_paths: dict[str, Path],
    threshold: float,
    character_threshold: float,
    replace_underscore: bool = True,
    trailing_comma: bool = False,
    exclude_tags: str = "",
    drop_overlap: bool = False,
    drop_blacklist: bool = False,
    drop_basic_character: bool = False,
    sort_mode: SortMode = "original",
    prioritize_people_tags: bool = True,
    pbar: ProgressBarLike | None = None,
) -> list[str]:
    spec = get_model_spec(model)
    session = open_onnx_session(model_paths["model.onnx"])
    images_np = images_to_uint8(images)

    results = []
    for image_np in images_np:
        if spec.family == "wd14":
            output = get_wd14_tags(
                image_np,
                session,
                csv_path=model_paths["selected_tags.csv"],
                threshold=threshold,
                character_threshold=character_threshold,
                no_underline=replace_underscore,
                drop_overlap=drop_overlap,
            )
        elif spec.family == "pixai":
            output = get_pixai_tags(
                image_np,
                session,
                csv_path=model_paths["selected_tags.csv"],
                thresholds_path=model_paths.get("thresholds.csv"),
                preprocess_path=model_paths.get("preprocess.json"),
                threshold=threshold,
                character_threshold=character_threshold,
                no_underline=replace_underscore,
                drop_overlap=drop_overlap,
            )
        else:
            raise ValueError(f"Unsupported tagger family: {spec.family}")

        tags = _ordered_tags_from_output(
            output,
            exclude_tags=exclude_tags,
            drop_blacklist=drop_blacklist,
            drop_basic_character=drop_basic_character,
            sort_mode=sort_mode,
            prioritize_people_tags=prioritize_people_tags,
        )
        results.append(
            tags_to_text(
                tags,
                use_spaces=False,
                use_escape=True,
                include_score=False,
                score_descend=False,
                trailing_comma=trailing_comma,
            )
        )
        _update_progress(pbar)

    return results
