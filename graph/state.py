"""LoanState schema — single source of truth for the graph state."""

from typing import TypedDict, Any, List, Dict, Optional, Union


class LoanState(TypedDict):
    """Flat state dict for the 14-step loan journey."""

    # Identity
    user_id: str

    # Journey tracking
    current_step: int           # 1–14
    max_steps_guard: int        # Incremented every node; terminate if > 25
    finished: bool

    # Collected data  {step_name: value}
    journey_data: Dict[str, Any]

    # Document lifecycle
    documents_uploaded: Dict[str, str]    # {doc_type: file_path}
    documents_status: Dict[str, str]      # {doc_type: pending|processing|completed|verified}
    extracted_data: Dict[str, Any]        # {doc_type: {field: value}}
    verification_queue: List[str]         # ["bank_statement", …]


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
