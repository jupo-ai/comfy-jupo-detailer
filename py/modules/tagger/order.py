import random
import re
from collections.abc import Mapping
from typing import Literal


SortMode = Literal["original", "shuffle", "score"]


def sort_tags(
    tags: list[str] | Mapping[str, float],
    mode: SortMode = "score",
    prioritize_people_tags: bool = True,
) -> list[str]:
    if mode not in {"original", "shuffle", "score"}:
        raise ValueError(f"Unknown sort_mode, 'original', 'shuffle' or 'score' expected but {mode!r} found.")

    npeople_tags = []
    remaining_tags = []

    if prioritize_people_tags and "solo" in tags:
        npeople_tags.append("solo")

    for tag in tags:
        if prioritize_people_tags and tag == "solo":
            continue
        if prioritize_people_tags and re.fullmatch(r"^\d+\+?(boy|girl)s?$", tag):
            npeople_tags.append(tag)
        else:
            remaining_tags.append(tag)

    if mode == "score":
        if isinstance(tags, Mapping):
            remaining_tags = sorted(remaining_tags, key=lambda x: -tags[x])
        else:
            raise TypeError(f"Sort mode {mode!r} not supported for list, for it do not have scores.")
    elif mode == "shuffle":
        random.shuffle(remaining_tags)

    return npeople_tags + remaining_tags
