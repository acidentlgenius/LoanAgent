"""LoanState schema — single source of truth for the graph state."""

from typing import TypedDict, Optional


class LoanState(TypedDict):
    """Flat state dict for the 15-step loan journey."""

    # Identity
    user_id: str

    # Journey tracking
    current_step: int           # 1–15
    max_steps_guard: int        # Incremented every node; terminate if > 25
    finished: bool

    # Collected data  {step_name: value}
    journey_data: dict

    # Document lifecycle
    documents_uploaded: dict    # {doc_type: file_path}
    documents_status: dict      # {doc_type: pending|processing|completed|verified}
    extracted_data: dict        # {doc_type: {field: value}}
    verification_queue: list    # ["bank_statement", …]


def initial_state(user_id: str) -> LoanState:
    """Factory — returns a clean starting state."""
    return LoanState(
        user_id=user_id,
        current_step=1,
        max_steps_guard=0,
        finished=False,
        journey_data={},
        documents_uploaded={},
        documents_status={},
        extracted_data={},
        verification_queue=[],
    )
