"""Indonesian informal-text preprocessing pipeline (FR-PRE-01 … FR-PRE-14)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields

from .exceptions import InvalidInputError
from .resources import load_json, load_lines

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
_RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#(\w+)")
_RE_NON_ALPHA = re.compile(r"[^A-Za-z\s]")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_REPEATED = re.compile(r"(.)\1{2,}")
_RE_TOKEN = re.compile(r"[A-Za-z]+")


@dataclass
class PreprocessConfig:
    """Toggleable preprocessing stages (FR-PRE-09, FR-PRE-12)."""

    cleaning: bool = True
    case_folding: bool = True
    reduce_repeated: bool = True
    slang: bool = True
    spell_correction: bool = True
    stopwords: bool = True
    max_edit_distance: int = 2
    min_token_length: int = 4
    max_tokens: int = 2000

    @classmethod
    def from_value(cls, value: bool | dict | PreprocessConfig | None) -> PreprocessConfig:
        """Build a config from ``True``/``False``/``dict``/instance."""
        if isinstance(value, cls):
            return value
        if value is None or value is True:
            return cls()
        if value is False:
            return cls(
                cleaning=False,
                case_folding=False,
                reduce_repeated=False,
                slang=False,
                spell_correction=False,
                stopwords=False,
            )
        if isinstance(value, Mapping):
            valid = {field.name for field in fields(cls)}
            unknown = set(value) - valid
            if unknown:
                raise InvalidInputError(
                    "Kunci konfigurasi preprocessing tidak dikenal: "
                    + ", ".join(sorted(unknown))
                )
            return cls(**value)
        raise InvalidInputError("Konfigurasi preprocessing harus bool, dict, atau PreprocessConfig.")


def damerau_levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    """Compute the Damerau-Levenshtein distance between ``a`` and ``b``.

    The implementation uses the optimal string alignment variant, which supports
    insertion, deletion, substitution, and transposition of adjacent characters.
    When ``max_distance`` is given, the function returns ``max_distance + 1`` as
    soon as the distance is known to exceed the threshold.
    """
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if max_distance is not None and abs(len_a - len_b) > max_distance:
        return max_distance + 1
    if not a:
        return len_b
    if not b:
        return len_a

    previous_previous = list(range(len_b + 1))
    previous = list(range(len_b + 1))

    for i in range(1, len_a + 1):
        current = [i] + [0] * len_b
        row_minimum = current[0]
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + cost,   # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                current[j] = min(current[j], previous_previous[j - 2] + 1)
            row_minimum = min(row_minimum, current[j])
        if max_distance is not None and row_minimum > max_distance:
            return max_distance + 1
        previous_previous, previous = previous, current

    return previous[len_b]


def reduce_repeated_chars(text: str) -> str:
    """Collapse runs of three or more identical characters (``bagussss`` → ``bagus``)."""
    return _RE_REPEATED.sub(r"\1", text)


def deletion_variants(word: str, max_distance: int) -> set[str]:
    """Return every string obtained by deleting up to ``max_distance`` characters.

    This is the SymSpell trick: two words within a small edit distance always
    share at least one deletion variant, so candidate lookup becomes a handful
    of dictionary probes instead of a full vocabulary scan.
    """
    results = {word}
    frontier = {word}
    for _ in range(max(0, max_distance)):
        next_frontier: set[str] = set()
        for item in frontier:
            for index in range(len(item)):
                next_frontier.add(item[:index] + item[index + 1 :])
        results |= next_frontier
        frontier = next_frontier
    return results


class TextPreprocessor:
    """Normalise Indonesian informal text.

    Parameters
    ----------
    config:
        A :class:`PreprocessConfig`, a dict of overrides, or a bool.
    slang_dict, stopwords, vocabulary, negation_words, intensifiers:
        Optional injections used mainly by tests. When omitted the packaged
        resources are loaded and cached.
    """

    def __init__(
        self,
        config: bool | dict | PreprocessConfig | None = None,
        *,
        slang_dict: Mapping[str, str] | None = None,
        stopwords: Iterable[str] | None = None,
        vocabulary: Iterable[str] | None = None,
        negation_words: Iterable[str] | None = None,
        intensifiers: Mapping[str, float] | None = None,
    ) -> None:
        self.config = PreprocessConfig.from_value(config)
        self.slang_dict = dict(slang_dict) if slang_dict is not None else dict(load_json("slang_dict.json"))
        self.stopwords = frozenset(stopwords) if stopwords is not None else frozenset(load_lines("stopwords_id.txt"))
        if vocabulary is not None:
            self._vocabulary_order: tuple[str, ...] = tuple(
                dict.fromkeys(str(word).lower() for word in vocabulary)
            )
        else:
            self._vocabulary_order = load_lines("vocab_id.txt")
        self.vocabulary = frozenset(self._vocabulary_order)
        self._rank = {word: index for index, word in enumerate(self._vocabulary_order)}
        self.negation_words = (
            frozenset(negation_words) if negation_words is not None else frozenset(load_lines("negation.txt"))
        )
        self.intensifiers = (
            dict(intensifiers) if intensifiers is not None else dict(load_json("intensifiers.json"))
        )
        self._index: dict[str, tuple[str, ...]] | None = None
        self._corrections: dict[str, str | None] = {}

    # -- individual stages -------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Remove URLs, mentions, numbers, punctuation and symbols."""
        if not isinstance(text, str):
            raise InvalidInputError("Teks harus berupa str.")
        cleaned = _RE_URL.sub(" ", text)
        cleaned = _RE_MENTION.sub(" ", cleaned)
        cleaned = _RE_HASHTAG.sub(r"\1", cleaned)
        cleaned = _RE_NON_ALPHA.sub(" ", cleaned)
        cleaned = _RE_WHITESPACE.sub(" ", cleaned).strip()
        if self.config.case_folding:
            cleaned = cleaned.lower()
        return cleaned

    def reduce_repeated(self, text: str) -> str:
        """Collapse irregular character repetition."""
        return reduce_repeated_chars(text)

    def tokenize(self, text: str) -> list[str]:
        """Split text into word tokens."""
        return [match.group() for match in _RE_TOKEN.finditer(text)]

    def tokenize_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into ``(token, start, end)`` triples."""
        return [(m.group(), m.start(), m.end()) for m in _RE_TOKEN.finditer(text)]

    def normalize_slang(self, tokens: Sequence[str]) -> list[str]:
        """Expand slang/abbreviations using the bundled dictionary."""
        expanded: list[str] = []
        for token in tokens:
            replacement = self.slang_dict.get(token)
            if replacement is None and token.endswith("nya") and len(token) > 5:
                replacement = self.slang_dict.get(token[:-3])
            if replacement is None:
                expanded.append(token)
            else:
                expanded.extend(part for part in replacement.split() if part)
        return expanded

    def spell_correction(self, tokens: Sequence[str]) -> list[str]:
        """Correct out-of-vocabulary tokens with Damerau-Levenshtein distance.

        Correction is skipped when case folding is disabled: the vocabulary is
        lowercase, so correcting mixed-case tokens would silently fold case
        against the user's configuration.
        """
        if not self.config.spell_correction or not self.config.case_folding:
            return list(tokens)
        corrected: list[str] = []
        for token in tokens:
            if (
                len(token) < self.config.min_token_length
                or token in self.vocabulary
                or token in self.slang_dict
                or token in self.negation_words
                or token in self.intensifiers
            ):
                corrected.append(token)
                continue
            candidate = self._correct_token(token)
            corrected.append(candidate if candidate is not None else token)
        return corrected

    def remove_stopwords(self, tokens: Sequence[str]) -> list[str]:
        """Remove stop words while preserving negations and intensifiers."""
        return [
            token
            for token in tokens
            if token not in self.stopwords
            or token in self.negation_words
            or token in self.intensifiers
        ]

    # -- full pipeline -----------------------------------------------------
    def normalize(self, text: str) -> str:
        """Run cleaning → case folding → repetition → slang → spelling."""
        cleaned = self.clean_text(text)
        if self.config.reduce_repeated:
            cleaned = reduce_repeated_chars(cleaned)
        tokens = self.tokenize(cleaned)
        if self.config.slang:
            tokens = self.normalize_slang(tokens)
        if self.config.spell_correction:
            tokens = self.spell_correction(tokens)
        return " ".join(tokens[: self.config.max_tokens])

    def process(self, text: str) -> str:
        """Run the full pipeline, including stop-word removal."""
        tokens = self.process_tokens(text)
        return self._finish(tokens)

    def process_normalized(self, text: str) -> str:
        """Apply tokenisation and stop-word removal to already-normalised text.

        The facade calls this on clause segments that were cut from the output
        of :meth:`normalize`, so the expensive stages are not repeated.
        """
        return self._finish(self.tokenize(text))

    def _finish(self, tokens: Sequence[str]) -> str:
        tokens = list(tokens[: self.config.max_tokens])
        if self.config.stopwords:
            tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)

    def process_tokens(self, text: str) -> list[str]:
        """Return the normalised token list (stop words included)."""
        cleaned = self.clean_text(text)
        if self.config.reduce_repeated:
            cleaned = reduce_repeated_chars(cleaned)
        tokens = self.tokenize(cleaned)
        if self.config.slang:
            tokens = self.normalize_slang(tokens)
        if self.config.spell_correction:
            tokens = self.spell_correction(tokens)
        return tokens[: self.config.max_tokens]

    def process_batch(self, texts: Iterable[str]) -> list[str]:
        """Apply :meth:`process` to every element of ``texts``."""
        return [self.process(text) for text in texts]

    # -- internals ---------------------------------------------------------
    def _build_index(self) -> dict[str, tuple[str, ...]]:
        """Index every vocabulary word by its deletion variants."""
        index: dict[str, list[str]] = {}
        distance = self.config.max_edit_distance
        for word in self._vocabulary_order:
            if len(word) > 20:  # pragma: no cover - pathological tokens
                continue
            for variant in deletion_variants(word, distance):
                index.setdefault(variant, []).append(word)
        return {variant: tuple(words) for variant, words in index.items()}

    def _correct_token(self, token: str) -> str | None:
        if token in self._corrections:
            return self._corrections[token]
        if len(token) > 20 or len(token) < self.config.min_token_length:
            self._corrections[token] = None
            return None
        if self._index is None:
            self._index = self._build_index()
        max_distance = self.config.max_edit_distance
        best: tuple[int, int, str] | None = None
        seen: set[str] = set()
        for variant in deletion_variants(token, max_distance):
            for word in self._index.get(variant, ()):
                if word in seen:
                    continue
                seen.add(word)
                if abs(len(word) - len(token)) > max_distance:
                    continue
                distance = damerau_levenshtein(token, word, max_distance)
                if distance == 0 or distance > max_distance:
                    continue
                # Ties are broken by corpus frequency (rank 0 = most frequent).
                candidate = (distance, self._rank.get(word, len(self._rank)), word)
                if best is None or candidate < best:
                    best = candidate
        result = best[2] if best is not None else None
        self._corrections[token] = result
        return result
