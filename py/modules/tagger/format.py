import re
from collections.abc import Mapping


RE_SPECIAL = re.compile(r"([\\()])")
_KAOMOJIS = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}


def add_underline(tag: str) -> str:
    return tag.strip().replace(" ", "_")


def remove_underline(tag: str) -> str:
    tag = tag.strip()
    return tag.replace("_", " ") if tag not in _KAOMOJIS else tag


def escape_tag(tag: str) -> str:
    return re.sub(RE_SPECIAL, r"\\\1", tag)


def tags_to_text(
    tags: Mapping[str, float],
    use_spaces: bool = False,
    use_escape: bool = True,
    include_score: bool = False,
    score_descend: bool = True,
    trailing_comma: bool = False,
) -> str:
    tag_pairs = tags.items()
    if score_descend:
        tag_pairs = sorted(tag_pairs, key=lambda x: (-x[1], x[0]))

    text_items = []
    for tag, score in tag_pairs:
        tag_text = remove_underline(tag) if use_spaces else tag
        if use_escape:
            tag_text = escape_tag(tag_text)
        if include_score:
            tag_text = f"({tag_text}:{score:.3f})"
        text_items.append(tag_text)

    text = ", ".join(text_items)
    if trailing_comma and text:
        text += ", "
    return text
