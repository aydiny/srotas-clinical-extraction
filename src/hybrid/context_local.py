"""
Lightweight ConText-style assertion detection (no medspaCy dependency).

medspaCy wheels currently fail to build on Windows (quickumls resource paths).
This module implements the same flags we need for the hybrid pipeline using
classic trigger / termination rules over a sentence window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from spacy.language import Language
from spacy.tokens import Doc, Span


@dataclass(frozen=True)
class _Trigger:
    pattern: re.Pattern[str]
    direction: str  # "forward" | "backward" | "bidirectional"
    max_distance: int = 80  # characters


# Ordered loosely by specificity. Patterns match on lowercased sentence text.
NEGATION = [
    _Trigger(re.compile(r"\b(?:no(?:t)?|without|denies|denied|denying)\b"), "forward"),
    _Trigger(re.compile(r"\b(?:ruled out|rule out|r\/o|negative for|free of|absence of)\b"), "forward"),
    _Trigger(re.compile(r"\b(?:unlikely|no evidence of)\b"), "forward"),
]
HISTORICAL = [
    _Trigger(re.compile(r"\b(?:history of|h\/o|hx of|prior|previous(?:ly)?|past)\b"), "forward"),
    _Trigger(re.compile(r"\b(?:remote history|formerly)\b"), "forward"),
]
FAMILY = [
    _Trigger(re.compile(
        r"\b(?:family history|fhx|mother|father|sister|brother|"
        r"maternal|paternal|parent|parents|daughter|son|aunt|uncle)\b"
    ), "bidirectional"),
]
UNCERTAIN = [
    _Trigger(re.compile(r"\b(?:possible|possibly|probable|probably|likely|unlikely)\b"), "forward"),
    _Trigger(re.compile(r"\b(?:suspect(?:ed)?|suspicious for|may be|might be|concern for)\b"), "forward"),
    _Trigger(re.compile(r"\?"), "bidirectional", max_distance=40),
]
HYPOTHETICAL = [
    _Trigger(re.compile(r"\b(?:if|should|would|could)\b"), "forward"),
]
CONDITIONAL = [
    _Trigger(re.compile(r"\b(?:if|when|unless)\b"), "forward"),
]

TERMINATORS = re.compile(
    r"[.!?]|\b(?:but|however|although|though|except|secondary to)\b"
)

ATTR_RULES: list[tuple[str, list[_Trigger]]] = [
    ("is_negated", NEGATION),
    ("is_historical", HISTORICAL),
    ("is_family", FAMILY),
    ("is_uncertain", UNCERTAIN),
    ("is_hypothetical", HYPOTHETICAL),
    ("is_conditional", CONDITIONAL),
]


def _ensure_extensions() -> None:
    for name, _ in ATTR_RULES:
        if not Span.has_extension(name):
            Span.set_extension(name, default=False)
    if not Doc.has_extension("context_rules"):
        Doc.set_extension("context_rules", default="local_v1")


def _sentence_window(doc: Doc, ent: Span) -> tuple[str, int]:
    """Return (sentence_text_lower, ent_start_offset_within_sentence)."""
    sent = ent.sent
    return sent.text.lower(), ent.start_char - sent.start_char


def _triggered(sentence: str, ent_local_start: int, ent_local_end: int, trigger: _Trigger) -> bool:
    before = sentence[:ent_local_start]
    after = sentence[ent_local_end:]

    def _match_side(side: str, from_right: bool) -> bool:
        window = side[-trigger.max_distance :] if from_right else side[: trigger.max_distance]
        # Ignore triggers beyond a terminator closer to the entity.
        if from_right:
            parts = TERMINATORS.split(window)
            window = parts[-1] if parts else window
        else:
            parts = TERMINATORS.split(window)
            window = parts[0] if parts else window
        return trigger.pattern.search(window) is not None

    if trigger.direction == "forward":
        return _match_side(before, from_right=True)
    if trigger.direction == "backward":
        return _match_side(after, from_right=False)
    return _match_side(before, from_right=True) or _match_side(after, from_right=False)


@Language.component("hybrid_context")
def hybrid_context(doc: Doc) -> Doc:
    _ensure_extensions()
    for ent in doc.ents:
        sentence, local_start = _sentence_window(doc, ent)
        local_end = local_start + len(ent.text)
        for attr, triggers in ATTR_RULES:
            value = any(
                _triggered(sentence, local_start, local_end, trig) for trig in triggers
            )
            setattr(ent._, attr, value)
    return doc


def add_local_context(nlp) -> None:
    """Attach the local ConText component if not already present."""
    _ensure_extensions()
    if "hybrid_context" not in nlp.pipe_names:
        nlp.add_pipe("hybrid_context", last=True)
