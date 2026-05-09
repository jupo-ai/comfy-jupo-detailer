import re
from functools import lru_cache


@lru_cache(maxsize=2048)
def _cached_singular_form(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")) and len(word) > 2:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word


@lru_cache(maxsize=2048)
def _cached_plural_form(word: str) -> str:
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{word}es"
    return f"{word}s"


def _split_to_words(text: str) -> list[str]:
    return [word.lower() for word in re.split(r"[\s_]+", text) if word]


def _words_to_matcher(words: list[str], enable_forms: bool = True) -> set[tuple[str, ...]]:
    if words and enable_forms:
        words_tuples = [
            words,
            [*words[:-1], _cached_singular_form(words[-1])],
            [*words[:-1], _cached_plural_form(words[-1])],
        ]
    else:
        words_tuples = [words]
    return {tuple(item) for item in words_tuples}


def tag_match_suffix(text: str, suffix: str) -> bool:
    suffix_words = _split_to_words(suffix)
    text_words = _split_to_words(text)
    if not suffix_words:
        return True

    text_words = text_words[-len(suffix_words):]
    return bool(_words_to_matcher(text_words) & _words_to_matcher(suffix_words))


def tag_match_prefix(text: str, prefix: str) -> bool:
    prefix_words = _split_to_words(prefix)
    text_words = _split_to_words(text)
    if not prefix_words:
        return True

    text_words = text_words[:len(prefix_words)]
    return bool(
        _words_to_matcher(text_words, enable_forms=False)
        & _words_to_matcher(prefix_words, enable_forms=False)
    )


def tag_match_full(t1: str, t2: str) -> bool:
    t1_words = _split_to_words(t1)
    t2_words = _split_to_words(t2)
    return bool(_words_to_matcher(t1_words) & _words_to_matcher(t2_words))
