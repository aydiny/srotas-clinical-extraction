from typing import Literal
from pydantic import BaseModel, Field, model_validator

# 1. The Key-Value Attribute Model
class Attribute(BaseModel):
    name: str = Field(description="The name of the qualifier, e.g., 'stage', 'dosage', 'route', 'test_value'.")
    value: str = Field(description="The value of the qualifier, e.g., 'III', '50mg', 'oral', '120', 'normal', 'positive', 'negative'.")
    unit: str | None = Field(default=None, description="The unit of measurement, e.g., 'mg', 'ml', 'Cells/cum', 'Iu/l', 'mg/dl'.")

# 2. Source Provenance (Crucial for the "hallucination detection" requirement)
class Provenance(BaseModel):
    source_text_quote: str = Field(
        description="The EXACT verbatim quote from the text that proves this fact. Must not be altered."
    )
    source_document: str = Field(
        description = "name of the source document "
    )
    page_number: int = Field(
        description="The page number in the source document."
    )

# 3. The Core Fact Model
class ClinicalFact(BaseModel):
    fact_id: str = Field(
        description="A unique identifier for this fact (e.g., 'fact_1') so other facts can reference it."
    )
    category: Literal["condition", "symptom", "observation", "medication", "procedure"] = Field(
        description="The type of clinical entity. Use 'observation' for labs, vitals, and social history."
    )
    name: str = Field(
        description="The primary name of the condition, medication, test, or procedure (e.g., 'Non-small cell lung cancer')."
    )

    # Contextual Qualifiers requested by the assessment
    status: Literal["active", "historical", "resolved", "unknown"] = Field(
        description="The current vs historical status of the entity."
    )
    is_negated: bool = Field(
        description="True if the text explicitly states the patient DOES NOT have this or it was ruled out."
    )
    certainty: Literal["confirmed", "suspected", "unknown"] = Field(
        description="Certainty of the finding. Use 'confirmed' for definitive findings/diagnoses, 'suspected' for possible/rule-out/working diagnoses, and 'unknown' if unclear."
    )

    temporality: Literal[
        "current",
        "historical",
        "at_encounter",
        "future_plan",
        "unknown"
    ] = Field(
        description="The temporal context of the fact."
    )

    date:  str | None = Field(
        default=None,
        description="The date of the fact."
    )

    # Dynamic Attributes & Qualifiers
    attributes: list[Attribute] = Field(
        default_factory=list,
        description="Specific qualifiers like stage, dosage, or test values."
    )

    # Relationships
    related_to_fact_ids: list[str] = Field(
        default_factory=list,
        description="A list of fact_ids this fact is related to (e.g., linking a medication to the condition it treats)."
    )

    # Provenance
    provenance: list[Provenance] = Field(
        default_factory=list,
        description="Evidence quotes supporting this extraction."
    )

    @model_validator(mode='after')
    def validate_temporality_and_status(self) -> 'ClinicalFact':
        # 'self' is the fully populated model instance
        if self.temporality == "historical" and self.status == "active":
            raise ValueError(
                "Contradiction detected: A 'historical' event cannot have an 'active' status. "
                "If the condition is ongoing, set temporality to 'current'."
            )
        return self

# 4. The Top-Level Representation
class PatientRepresentation(BaseModel):
    patient_id: str | None = Field(
        default=None,
        description="The unique enterprise identifier for the patient (e.g., MRN)."
    )
    patient_name: str | None = Field(
        default=None,
        description="The patient's name. WARNING: In a production pipeline, this PHI field must be stripped, masked, or pseudonymized before the structured representation is persisted to the analytics database."
    )
    age: int | None = Field(default=None)
    sex: Literal["Female", "Male", "Other", "Unknown"] = Field(default="Unknown")
    clinical_facts: list[ClinicalFact] = Field(
        default_factory=list,
        description="The longitudinal history of patient facts extracted from the record."
    )

class CandidateFact(BaseModel):
    candidate_id: str = Field(
        description="A unique identifier for this candidate fact (e.g., 'candidate_1') so other candidate facts can reference it."
    )
    category_hint: Literal[
        "condition",
        "symptom",
        "observation",
        "medication",
        "procedure",
        "unknown",
    ]
    raw_name: str = Field(
        description="The raw name of the candidate fact (e.g., 'Non-small cell lung cancer')."
    )
    page_number: int = Field(
        description="The page number in the source document."
    )
    evidence_text: str = Field(
        description="The evidence text that supports this candidate fact."
    )


class CandidateFactInventory(BaseModel):
    candidates: list[CandidateFact]