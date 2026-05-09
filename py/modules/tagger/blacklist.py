from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import requests

from .match import _split_to_words, _words_to_matcher


TAG_FILTERING_REPO = "alea31415/tag_filtering"
BLACKLIST_FILENAME = "blacklist_tags.txt"
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
def _load_online_blacklist() -> list[str]:
    with open(_download_dataset_file(BLACKLIST_FILENAME), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@lru_cache(maxsize=1)
def _online_blacklist_set() -> set[tuple[str, ...]]:
    set_ = set()
    for tag in _load_online_blacklist():
        set_ = set_ | _words_to_matcher(_split_to_words(tag))
    return set_


def _is_blacklisted(tag: str, blacklist_set: set[tuple[str, ...]]) -> bool:
    tag_matcher = _words_to_matcher(_split_to_words(tag))
    return bool(tag_matcher & blacklist_set)


def is_blacklisted(tag: str) -> bool:
    return _is_blacklisted(tag, _online_blacklist_set())


def drop_blacklisted_tags(
    tags: list[str] | Mapping[str, float],
    use_presets: bool = True,
    custom_blacklist: list[str] | None = None,
) -> list[str] | dict[str, float]:
    blacklist = set()
    if use_presets:
        blacklist = blacklist | _online_blacklist_set()
    for tag in custom_blacklist or []:
        blacklist = blacklist | _words_to_matcher(_split_to_words(tag))

    if isinstance(tags, Mapping):
        return {tag: value for tag, value in tags.items() if not _is_blacklisted(tag, blacklist)}
    if isinstance(tags, list):
        return [tag for tag in tags if not _is_blacklisted(tag, blacklist)]
    raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")
