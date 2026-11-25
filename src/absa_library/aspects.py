"""General (open-domain) aspect extraction and clause segmentation.

The extractor never assumes a fixed domain taxonomy. In ``auto`` mode aspects
are discovered from the text by pairing candidate noun phrases with opinion
terms found in the sentiment lexicon. A user-supplied dictionary can still be
provided for controlled experiments.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .exceptions import InvalidInputError
from .lexicon import SentimentLexicon
from .preprocessing import TextPreprocessor
from .resources import load_lines

_TOKEN_RE = re.compile(r"[A-Za-z]+")

CLAUSE_MARKERS = frozenset(
    {
        "tapi",
        "tetapi",
        "namun",
        "sedangkan",
        "melainkan",
        "meskipun",
        "walaupun",
        "walau",
        "padahal",
        "sementara",
        "sebaliknya",
    }
)

# Tokens that are never aspect candidates.
_NON_ASPECT = frozenset(
    {
        "yang",
        "itu",
        "ini",
        "saya",
        "aku",
        "kami",
        "kita",
        "dia",
        "mereka",
        "anda",
        "kamu",
        "nya",
        "juga",
        "sudah",
        "telah",
        "akan",
        "masih",
        "bisa",
        "dapat",
        "ada",
        "tidak",
        "bukan",
        "belum",
        "jangan",
        "kurang",
        "sangat",
        "banget",
        "sekali",
        "paling",
        "agak",
        "lebih",
        "saja",
        "sih",
        "deh",
        "kok",
        "ya",
        "yah",
        "dong",
        "nih",
        "loh",
        "lah",
        "kah",
        "pun",
    }
)

_FALLBACK_ASPECT = "UMUM"


@dataclass(frozen=True)
class AspectMention:
    """One aspect occurrence inside a text."""

    aspect: str
    surface: str
    start: int
    end: int
    source: str = "auto"


class AspectExtractor:
    """Extract aspect mentions from normalised Indonesian text.

    Parameters
    ----------
    aspects:
        ``"auto"`` (default) for open-domain extraction, a mapping of
        ``{label: [keywords]}`` for dictionary mode, or ``None``/``"none"`` to
        disable aspect extraction.
    aspect_vocab:
        Optional iterable of general aspect nouns used to boost precision.
    synonyms:
        Optional mapping from surface form to canonical label.
    window:
        Maximum token distance between an aspect and an opinion term.
    max_aspects:
        Hard cap on the number of aspects returned per text.
    """

    def __init__(
        self,
        aspects: str | Mapping[str, Iterable[str]] | None = "auto",
        *,
        aspect_vocab: Iterable[str] | None = None,
        synonyms: Mapping[str, str] | None = None,
        window: int = 5,
        max_aspects: int = 10,
        lexicon: SentimentLexicon | None = None,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        if isinstance(aspects, str) and aspects.lower() not in {"auto", "none"}:
            raise InvalidInputError("Mode aspects harus 'auto', 'none', None, atau dict.")
        if aspects is not None and not isinstance(aspects, (str, Mapping)):
            if isinstance(aspects, (list, tuple, set)):
                aspects = {str(item): [str(item)] for item in aspects}
            else:
                raise InvalidInputError("Parameter aspects harus 'auto', dict, list, atau None.")

        self.mode = "auto"
        if aspects is None:
            self.mode = "none"
        elif isinstance(aspects, str) and aspects.lower() == "none":
            self.mode = "none"
        elif isinstance(aspects, Mapping):
            self.mode = "dictionary"

        self.dictionary: dict[str, list[tuple[str, ...]]] = {}
        if self.mode == "dictionary":
            for label, keywords in aspects.items():  # type: ignore[union-attr]
                phrases = [self._phrase_tokens(keyword) for keyword in keywords]
                self.dictionary[str(label)] = [phrase for phrase in phrases if phrase]

        self.preprocessor = preprocessor or TextPreprocessor()
        self.lexicon = lexicon or SentimentLexicon()
        self.synonyms = dict(synonyms or {})
        self.window = max(1, int(window))
        self.max_aspects = max(1, int(max_aspects))
        vocab = aspect_vocab if aspect_vocab is not None else load_lines("aspect_vocab.txt")
        self.aspect_vocab = frozenset(word.lower() for word in vocab if " " not in word)

    # -- public API --------------------------------------------------------
    def extract(self, text: str) -> list[AspectMention]:
        """Return every aspect mention found in ``text``."""
        if not isinstance(text, str):
            raise InvalidInputError("Teks untuk ekstraksi aspek harus berupa str.")
        tokens = self._tokenize(text)
        if not tokens:
            return []
        if self.mode == "none":
            return []
        if self.mode == "dictionary":
            mentions = self._extract_dictionary(text, tokens)
        else:
            mentions = self._extract_auto(text, tokens)
        return mentions[: self.max_aspects]

    def segment(self, text: str, mention: AspectMention, *, max_clause_tokens: int = 40) -> str:
        """Return the clause (or window) that surrounds ``mention``."""
        tokens = self._tokenize(text)
        if not tokens:
            return text
        index = self._token_index_at(tokens, mention.start)
        if index is None:
            return text
        for start, end in self.clause_ranges(tokens):
            if start <= index < end:
                if end - start <= max_clause_tokens:
                    return text[tokens[start][1] : tokens[end - 1][2]]
                left = max(start, index - 15)
                right = min(end, index + 16)
                return text[tokens[left][1] : tokens[right - 1][2]]
        return text

    def segment_for_label(self, text: str, aspect: str) -> str:
        """Return the segment for a known aspect label, falling back to the full text."""
        mentions = self.extract(text)
        wanted = aspect.strip().lower()
        for mention in mentions:
            if mention.aspect.lower() == wanted:
                return self.segment(text, mention)
        # The label may be present in the text even if auto-extraction missed it.
        for token, start, end in self._tokenize(text):
            if token.lower() == wanted:
                synthetic = AspectMention(aspect, token, start, end, "manual")
                return self.segment(text, synthetic)
        return text

    def clause_ranges(self, tokens: Sequence[tuple[str, int, int]]) -> list[tuple[int, int]]:
        """Return token index ranges for every clause."""
        if not tokens:
            return []
        starts = [0]
        for index, (token, _, _) in enumerate(tokens):
            if index > 0 and token in CLAUSE_MARKERS:
                starts.append(index)
        starts = sorted(set(starts))
        ranges: list[tuple[int, int]] = []
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(tokens)
            if end > start:
                ranges.append((start, end))
        return ranges

    # -- extraction strategies --------------------------------------------
    def _extract_dictionary(
        self, text: str, tokens: Sequence[tuple[str, int, int]]
    ) -> list[AspectMention]:
        mentions: list[AspectMention] = []
        words = [token.lower() for token, _, _ in tokens]
        for label, phrases in self.dictionary.items():
            for phrase in phrases:
                length = len(phrase)
                for start in range(0, len(words) - length + 1):
                    if tuple(words[start : start + length]) == phrase:
                        mentions.append(
                            AspectMention(
                                aspect=label,
                                surface=text[tokens[start][1] : tokens[start + length - 1][2]],
                                start=tokens[start][1],
                                end=tokens[start + length - 1][2],
                                source="dictionary",
                            )
                        )
        return self._deduplicate(mentions)

    def _extract_auto(
        self, text: str, tokens: Sequence[tuple[str, int, int]]
    ) -> list[AspectMention]:
        words = [token.lower() for token, _, _ in tokens]
        opinions = {
            index
            for index, word in enumerate(words)
            if self.lexicon.score(word) != 0.0
        }
        mentions: list[AspectMention] = []
        covered: set[int] = set()

        for opinion_index in sorted(opinions):
            candidate = self._nearest_candidate(words, opinion_index)
            if candidate is None:
                continue
            index, label = candidate
            if index in covered:
                continue
            covered.add(index)
            mentions.append(
                AspectMention(
                    aspect=self._canonical(label),
                    surface=text[tokens[index][1] : tokens[index][2]],
                    start=tokens[index][1],
                    end=tokens[index][2],
                    source="auto",
                )
            )

        # Vocabulary hits without a nearby opinion are still useful: they will be
        # classified as Netral by the engine.
        for index, word in enumerate(words):
            if index in covered or index in opinions:
                continue
            if word in self.aspect_vocab and self._is_aspect_candidate(word):
                covered.add(index)
                mentions.append(
                    AspectMention(
                        aspect=self._canonical(word),
                        surface=text[tokens[index][1] : tokens[index][2]],
                        start=tokens[index][1],
                        end=tokens[index][2],
                        source="vocab",
                    )
                )
        return self._deduplicate(mentions)

    def _nearest_candidate(
        self, words: Sequence[str], opinion_index: int
    ) -> tuple[int, str] | None:
        best_vocab: tuple[int, int, str] | None = None
        best_any: tuple[int, int, str] | None = None
        for distance in range(1, self.window + 1):
            for index in (opinion_index - distance, opinion_index + distance):
                if index < 0 or index >= len(words):
                    continue
                word = words[index]
                if not self._is_aspect_candidate(word):
                    continue
                candidate = (distance, index, word)
                if word in self.aspect_vocab:
                    if best_vocab is None or candidate < best_vocab:
                        best_vocab = candidate
                elif best_any is None or candidate < best_any:
                    best_any = candidate
            if best_vocab is not None:
                break
        chosen = best_vocab or best_any
        if chosen is None:
            return None
        return chosen[1], chosen[2]

    def _is_aspect_candidate(self, word: str) -> bool:
        if len(word) < 3 or word in _NON_ASPECT:
            return False
        if word in self.preprocessor.stopwords:
            return False
        if word in self.preprocessor.negation_words or word in self.preprocessor.intensifiers:
            return False
        if word in CLAUSE_MARKERS:
            return False
        if self.lexicon.score(word) != 0.0:
            return False
        return True

    def _canonical(self, label: str) -> str:
        label = label.strip().lower()
        if label.endswith("nya") and len(label) > 5:
            label = label[:-3]
        return self.synonyms.get(label, label)

    def _deduplicate(self, mentions: Iterable[AspectMention]) -> list[AspectMention]:
        seen: set[str] = set()
        unique: list[AspectMention] = []
        for mention in mentions:
            key = mention.aspect.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(mention)
        return unique

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[tuple[str, int, int]]:
        return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]

    @staticmethod
    def _phrase_tokens(phrase: str) -> tuple[str, ...]:
        return tuple(token.lower() for token in _TOKEN_RE.findall(str(phrase)))

    @staticmethod
    def _token_index_at(tokens: Sequence[tuple[str, int, int]], position: int) -> int | None:
        for index, (_, start, end) in enumerate(tokens):
            if start <= position < end:
                return index
        if position <= tokens[0][1]:
            return 0
        return None


def fallback_mention(text: str) -> AspectMention:
    """Return the sentence-level ``UMUM`` mention used when nothing is detected."""
    return AspectMention(_FALLBACK_ASPECT, text, 0, len(text), "fallback")
