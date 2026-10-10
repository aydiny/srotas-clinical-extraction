# evaluate.py
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from validate import norm


# ---------- Matching configuration ----------

ALIGNMENT_THRESHOLD = 0.55
MIN_NAME_SIMILARITY = 0.50
ATTRIBUTE_VALUE_THRESHOLD = 85

CONTEXT_FIELDS = [
    "status",
    "temporality",
    "certainty",
    "is_negated",
]


# ---------- Matching helpers ----------

def quote_tokens(fact) -> set[str]:
    """
    Tokenise all provenance quotations associated with a fact.

    Uses a set because this is Jaccard overlap: repeated words should
    not get extra weight.
    """
    text = " ".join(
        norm(provenance.source_text_quote)
        for provenance in fact.provenance
    )
    return set(text.split())


def pair_score(pred_fact, gold_fact) -> float:
    """
    Score whether one predicted fact represents one gold fact.

    Fact identity is mainly name plus category. Provenance overlap helps
    resolve synonyms or different surface forms, but cannot by itself
    match two unrelated facts from the same sentence.
    """
    name_score = (
        fuzz.token_set_ratio(
            norm(pred_fact.name),
            norm(gold_fact.name),
        )
        / 100
    )

    # Prevent "Penicillin" from matching "Doxycycline" just because
    # both cite the same medication-list sentence.
    if name_score < MIN_NAME_SIMILARITY:
        return 0.0

    category_score = (
        1.0
        if pred_fact.category == gold_fact.category
        else 0.0
    )

    pred_tokens = quote_tokens(pred_fact)
    gold_tokens = quote_tokens(gold_fact)

    evidence_overlap = len(pred_tokens & gold_tokens) / max(len(pred_tokens | gold_tokens),1,)

    return (
        0.50 * name_score
        + 0.20 * category_score
        + 0.30 * evidence_overlap
    )


def align(pred_facts, gold_facts, threshold: float = ALIGNMENT_THRESHOLD):
    """
    Return one-to-one matches as:
        [(predicted_index, gold_index, similarity_score), ...]

    Hungarian assignment maximises the total similarity across all
    one-to-one pairings, then the threshold removes weak forced pairs.
    """
    if not pred_facts or not gold_facts:
        return []

    similarity_matrix = np.array(
        [
            [pair_score(pred, gold) for gold in gold_facts]
            for pred in pred_facts
        ]
    )

    predicted_indices, gold_indices = linear_sum_assignment(
        similarity_matrix,
        maximize=True,
    )

    return [
        (int(pred_index), int(gold_index), float(similarity_matrix[pred_index, gold_index]))
        for pred_index, gold_index in zip(predicted_indices, gold_indices)
        if similarity_matrix[pred_index, gold_index] >= threshold
    ]


# ---------- Attribute and context scoring ----------

def is_context_correct(
    field_name: str,
    predicted_value: Any,
    gold_fact,
    sidecar: dict,
) -> bool:
    """
    Compare context fields, allowing specifically documented alternatives
    stored in the optional sidecar file.
    """
    accepted = (
        sidecar
        .get(gold_fact.fact_id, {})
        .get("acceptable", {})
        .get(field_name, [getattr(gold_fact, field_name)])
    )

    return predicted_value in accepted


def attribute_matches(gold_attribute, predicted_attributes) -> bool:
    """
    Attribute names must match after normalisation. Values use fuzzy
    matching to avoid harmless punctuation/case differences.

    Units are compared only when the gold unit is populated.
    """
    for predicted_attribute in predicted_attributes:
        same_name = (
            norm(predicted_attribute.name)
            == norm(gold_attribute.name)
        )

        similar_value = (
            fuzz.ratio(
                norm(predicted_attribute.value),
                norm(gold_attribute.value),
            )
            >= ATTRIBUTE_VALUE_THRESHOLD
        )

        compatible_unit = (
            gold_attribute.unit is None
            or norm(predicted_attribute.unit or "")
            == norm(gold_attribute.unit)
        )

        if same_name and similar_value and compatible_unit:
            return True

    return False


# ---------- Main evaluation ----------

def evaluate_patient(pred, gold, reports, sidecar: dict | None = None):
    """
    Compare one extracted PatientRepresentation with one gold representation.

    Returns:
        counts: machine-readable values for aggregate metrics
        errors: flat rows suitable for error_table.csv
    """
    sidecar = sidecar or {}

    pred_facts = pred.clinical_facts
    gold_facts = gold.clinical_facts

    matches = align(pred_facts, gold_facts)

    matched_predicted = {
        pred_index: (gold_index, similarity)
        for pred_index, gold_index, similarity in matches
    }

    matched_gold = {
        gold_index: pred_index
        for pred_index, gold_index, _ in matches
    }

    # `reports` comes from validate.py.
    # supported=False means one or more validation errors.
    supported_by_fact_id = {
        report.fact_id: report.supported
        for report in reports
    }

    errors = []

    context_correct = Counter()
    attributes_total = 0
    attributes_correct = 0

    # Score matched facts: context and attributes.
    for pred_index, (gold_index, similarity) in matched_predicted.items():
        pred_fact = pred_facts[pred_index]
        gold_fact = gold_facts[gold_index]

        for field_name in CONTEXT_FIELDS:
            pred_value = getattr(pred_fact, field_name)

            if is_context_correct(
                field_name,
                pred_value,
                gold_fact,
                sidecar,
            ):
                context_correct[field_name] += 1
            else:
                errors.append(
                    {
                        "error_class": f"wrong_{field_name}",
                        "gold_fact_id": gold_fact.fact_id,
                        "pred_fact_id": pred_fact.fact_id,
                        "fact_name": gold_fact.name,
                        "gold_value": getattr(gold_fact, field_name),
                        "predicted_value": pred_value,
                        "similarity": round(similarity, 3),
                    }
                )

        for gold_attribute in gold_fact.attributes:
            attributes_total += 1

            if attribute_matches(
                gold_attribute,
                pred_fact.attributes,
            ):
                attributes_correct += 1
            else:
                errors.append(
                    {
                        "error_class": "wrong_or_missing_attribute",
                        "gold_fact_id": gold_fact.fact_id,
                        "pred_fact_id": pred_fact.fact_id,
                        "fact_name": gold_fact.name,
                        "gold_value": (
                            f"{gold_attribute.name}="
                            f"{gold_attribute.value}"
                        ),
                        "predicted_value": "",
                        "similarity": round(similarity, 3),
                    }
                )

        # A predicted fact can match the gold fact but still be unsupported.
        if not supported_by_fact_id.get(pred_fact.fact_id, False):
            errors.append(
                {
                    "error_class": "unsupported_matched_fact",
                    "gold_fact_id": gold_fact.fact_id,
                    "pred_fact_id": pred_fact.fact_id,
                    "fact_name": pred_fact.name,
                    "gold_value": "",
                    "predicted_value": "",
                    "similarity": round(similarity, 3),
                }
            )

    # Gold facts with no matching prediction are missed.
    for gold_index, gold_fact in enumerate(gold_facts):
        if gold_index not in matched_gold:
            is_critical = (
                sidecar
                .get(gold_fact.fact_id, {})
                .get("critical", False)
            )

            errors.append(
                {
                    "error_class": (
                        "missed_critical"
                        if is_critical
                        else "missed"
                    ),
                    "gold_fact_id": gold_fact.fact_id,
                    "pred_fact_id": "",
                    "fact_name": gold_fact.name,
                    "gold_value": gold_fact.category,
                    "predicted_value": "",
                    "similarity": "",
                }
            )

    # Predictions with no gold match need source validation before being
    # called false positives. Your gold set is intentionally incomplete.
    extra_supported = 0
    unsupported_unmatched = 0

    for pred_index, pred_fact in enumerate(pred_facts):
        if pred_index in matched_predicted:
            continue

        is_supported = supported_by_fact_id.get(
            pred_fact.fact_id,
            False,
        )

        if is_supported:
            extra_supported += 1
            error_class = "extra_supported"
        else:
            unsupported_unmatched += 1
            error_class = "unsupported"

        errors.append(
            {
                "error_class": error_class,
                "gold_fact_id": "",
                "pred_fact_id": pred_fact.fact_id,
                "fact_name": pred_fact.name,
                "gold_value": "",
                "predicted_value": pred_fact.category,
                "similarity": "",
            }
        )

    counts = {
        "predicted_facts": len(pred_facts),
        "gold_facts": len(gold_facts),
        "true_positives": len(matches),
        "false_negatives": len(gold_facts) - len(matches),
        "unmatched_predictions": (
            len(pred_facts) - len(matches)
        ),
        "extra_supported": extra_supported,
        "unsupported_unmatched": unsupported_unmatched,
        "unsupported_total": sum(
            not is_supported
            for is_supported in supported_by_fact_id.values()
        ),
        "context_correct": dict(context_correct),
        "attributes_total": attributes_total,
        "attributes_correct": attributes_correct,
    }

    return counts, errors


# ---------- Aggregate results across documents ----------

def aggregate(all_counts: list[dict]) -> dict:
    """
    Aggregate results across all documents in one run.
    """
    if not all_counts:
        return {}

    def total(key: str) -> int:
        return sum(count[key] for count in all_counts)

    total_matches = total("true_positives")
    total_gold = total("gold_facts")
    total_predicted = total("predicted_facts")
    total_attributes = total("attributes_total")

    context_accuracy = {}

    for field_name in CONTEXT_FIELDS:
        correctly_scored = sum(
            count["context_correct"].get(field_name, 0)
            for count in all_counts
        )

        context_accuracy[field_name] = (
            correctly_scored / max(total_matches, 1)
        )

    return {
        "gold_recall": total_matches / max(total_gold, 1),

        # This is intentionally not called "precision", because gold is
        # a selected subset rather than exhaustive annotation.
        "precision_vs_gold_subset": (
            total_matches / max(total_predicted, 1)
        ),

        # More meaningful where gold is incomplete: supported extractions
        # are not treated as errors.
        "precision_excluding_extra_supported": (
            total_matches
            / max(
                total_matches + total("unsupported_unmatched"),
                1,
            )
        ),

        "extra_supported_facts": total("extra_supported"),

        "unsupported_rate": (
            total("unsupported_total")
            / max(total_predicted, 1)
        ),

        "context_accuracy": context_accuracy,

        "attribute_recall_on_matched_facts": (
            total("attributes_correct")
            / max(total_attributes, 1)
        ),

        "counts": {
            "predicted_facts": total_predicted,
            "gold_facts": total_gold,
            "true_positives": total_matches,
            "false_negatives": total("false_negatives"),
            "unmatched_predictions": total(
                "unmatched_predictions"
            ),
        },
    }