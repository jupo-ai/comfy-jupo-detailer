import copy
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import requests


TAG_FILTERING_REPO = "alea31415/tag_filtering"
OVERLAP_FILENAME = "overlap_tags_simplified.json"
_HF_DATASET_BASE_URL = f"https://huggingface.co/datasets/{TAG_FILTERING_REPO}/resolve/main"
_CACHE_DIR = Path(__file__).parents[3] / "tagger_models" / "tag_filtering"


def _download_dataset_file(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / filename
    if path.exists():
        return path

    response = requests.get(f"{_HF_DATASET_BASE_URL}/{filename}", timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


@lru_cache(maxsize=1)
def _get_overlap_tags() -> Mapping[str, list[str]]:
    with open(_download_dataset_file(OVERLAP_FILENAME), "r", encoding="utf-8") as f:
        return json.load(f)


def drop_overlap_tags(tags: list[str] | Mapping[str, float]) -> list[str] | dict[str, float]:
    overlap_tags_dict = _get_overlap_tags()
    result_tags = []
    origin_tags = copy.deepcopy(tags)
    if isinstance(tags, Mapping):
        tags = list(tags.keys())
    elif not isinstance(tags, list):
        raise TypeError(f"Unknown tags type - {origin_tags!r}.")

    tags_underscore = [tag.replace(" ", "_") for tag in tags]
    tags_underscore_set = set(tags_underscore)

    for tag, tag_ in zip(tags, tags_underscore):
        to_remove = False

        if tag_ in overlap_tags_dict:
            overlap_values = set(overlap_tags_dict[tag_])
            if overlap_values.intersection(tags_underscore_set):
                to_remove = True

        for tag_another in tags:
            if tag in tag_another and tag != tag_another:
                to_remove = True
                break

        if not to_remove:
            result_tags.append(tag)

    if isinstance(origin_tags, list):
        return result_tags
    if isinstance(origin_tags, Mapping):
        result_tags_set = set(result_tags)
        return {key: value for key, value in origin_tags.items() if key in result_tags_set}
    raise TypeError(f"Unknown tags type - {origin_tags!r}.")
