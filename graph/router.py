"""Deterministic router — NO LLM calls, pure rule-based branching."""

from typing import Literal
from graph.state import LoanState
from config import MAX_STEPS_GUARD


# All valid destinations for add_conditional_edges
RouterDest = Literal[
    "verification_node",
    "step_1_name", "step_2_dob", "step_3_contact", "step_4_income",
    "step_5_document_upload", "step_6_employment", "step_7_address",
    "step_8_loan_amount", "step_9_loan_tenure", "step_10_purpose",
    "step_11_references", "step_12_bank_details", "step_13_consent",
    "step_14_review", "step_15_summary",
    "finish",
]

# Mapping: step number → node name
STEP_NODE_MAP: dict[int, str] = {
    1: "step_1_name",
    2: "step_2_dob",
    3: "step_3_contact",
    4: "step_4_income",
    5: "step_5_document_upload",
    6: "step_6_employment",
    7: "step_7_address",
    8: "step_8_loan_amount",
    9: "step_9_loan_tenure",
    10: "step_10_purpose",
    11: "step_11_references",
    12: "step_12_bank_details",
    13: "step_13_consent",
    14: "step_14_review",
    15: "step_15_summary",
}


def router(state: LoanState) -> RouterDest:
    """
    Rule-based router.  Priority: termination > verification > next step.
    Called via add_conditional_edges after every node.
    """
    # Guard: hard terminate if exceeded
    if state["max_steps_guard"] > MAX_STEPS_GUARD:
        return "finish"

    # Already done
    if state["finished"]:
        return "finish"

    # Priority 1: pending verifications
    if state["verification_queue"]:
        return "verification_node"

    # Priority 2: continue journey
    step = state["current_step"]
    if step in STEP_NODE_MAP:
        return STEP_NODE_MAP[step]

    # Default: done
    return "finish"
