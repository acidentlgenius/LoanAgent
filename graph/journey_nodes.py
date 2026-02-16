"""15 journey step nodes — LLM-powered, data-driven.

Each step:
  1. LLM generates a contextual, conversational prompt
  2. interrupt() pauses for user input
  3. LLM extracts structured data from free-text response
  4. State advances deterministically
"""

from langgraph.types import interrupt
from graph.state import LoanState
from graph.llm import (
    generate_step_message,
    extract_step_data,
    generate_review_summary,
    generate_final_summary,
)
from workers.document_processor import process_document_async

# ── Document types expected at step 5 ──────────────────────────────────
REQUIRED_DOCUMENTS = ["bank_statement", "payslip", "cibil", "pan", "aadhaar"]


# ── Core step handler (DRY) ────────────────────────────────────────────
def _llm_step(state: LoanState, step_name: str) -> dict:
    """
    Generic LLM-powered step:
      - Generate contextual prompt via LLM
      - Interrupt for user input
      - Extract structured data via LLM
      - Advance state
    """
    prompt = generate_step_message(step_name, state["current_step"], state["journey_data"])

    user_text = interrupt({
        "type": "journey_step",
        "step": state["current_step"],
        "field": step_name,
        "message": prompt,
    })

    # LLM extracts structured fields from free-text
    extracted = extract_step_data(step_name, str(user_text))

    return {
        "journey_data": {**state["journey_data"], step_name: extracted},
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


# ── Step functions ──────────────────────────────────────────────────────

def step_1_name(state: LoanState) -> dict:
    """Collect first name and last name."""
    return _llm_step(state, "name")


def step_2_dob(state: LoanState) -> dict:
    """Collect date of birth."""
    return _llm_step(state, "dob")


def step_3_contact(state: LoanState) -> dict:
    """Collect phone and email."""
    return _llm_step(state, "contact")


def step_4_income(state: LoanState) -> dict:
    """Collect monthly income and source."""
    return _llm_step(state, "income")


def step_5_document_upload(state: LoanState) -> dict:
    """Collect document uploads — kicks off async processing."""
    prompt = generate_step_message("document_upload", 5, state["journey_data"])

    user_input = interrupt({
        "type": "document_upload",
        "step": 5,
        "required_documents": REQUIRED_DOCUMENTS,
        "message": prompt,
    })

    # Parse uploads: user_input can be dict {doc_type: path} or free text
    uploads = user_input if isinstance(user_input, dict) else {}
    new_uploaded = {**state["documents_uploaded"], **uploads}
    new_status = {**state["documents_status"]}

    for doc_type, file_path in uploads.items():
        new_status[doc_type] = "processing"
        process_document_async(doc_type, file_path)

    return {
        "documents_uploaded": new_uploaded,
        "documents_status": new_status,
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


def step_6_employment(state: LoanState) -> dict:
    """Collect employer, designation, tenure."""
    return _llm_step(state, "employment")


def step_7_address(state: LoanState) -> dict:
    """Collect current address."""
    return _llm_step(state, "address")


def step_8_loan_amount(state: LoanState) -> dict:
    """Collect requested loan amount."""
    return _llm_step(state, "loan_amount")


def step_9_loan_tenure(state: LoanState) -> dict:
    """Collect loan tenure in months."""
    return _llm_step(state, "loan_tenure")


def step_10_purpose(state: LoanState) -> dict:
    """Collect loan purpose."""
    return _llm_step(state, "purpose")


def step_11_references(state: LoanState) -> dict:
    """Collect references / guarantor info."""
    return _llm_step(state, "references")


def step_12_bank_details(state: LoanState) -> dict:
    """Collect disbursement bank account."""
    return _llm_step(state, "bank_details")


def step_13_consent(state: LoanState) -> dict:
    """Collect T&C consent."""
    return _llm_step(state, "consent")


def step_14_review(state: LoanState) -> dict:
    """Present LLM-generated summary for review."""
    summary = generate_review_summary(state["journey_data"])

    user_input = interrupt({
        "type": "review",
        "step": 14,
        "collected_data": state["journey_data"],
        "documents_status": state["documents_status"],
        "message": f"📋 **Application Review**\n\n{summary}\n\nPlease confirm everything looks correct, or let me know what to change.",
    })

    return {
        "journey_data": {**state["journey_data"], "review_confirmed": str(user_input)},
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


def step_15_summary(state: LoanState) -> dict:
    """Final summary — LLM generates confirmation, marks journey finished."""
    final_msg = generate_final_summary(state["journey_data"], state["documents_status"])

    interrupt({
        "type": "summary",
        "step": 15,
        "journey_data": state["journey_data"],
        "documents_status": state["documents_status"],
        "message": final_msg,
    })

    return {
        "current_step": 16,
        "max_steps_guard": state["max_steps_guard"] + 1,
        "finished": True,
    }


# ── Registry (used by builder.py) ──────────────────────────────────────
STEP_FUNCTIONS: dict[int, callable] = {
    1: step_1_name,       2: step_2_dob,
    3: step_3_contact,    4: step_4_income,
    5: step_5_document_upload,
    6: step_6_employment, 7: step_7_address,
    8: step_8_loan_amount, 9: step_9_loan_tenure,
    10: step_10_purpose,  11: step_11_references,
    12: step_12_bank_details, 13: step_13_consent,
    14: step_14_review,   15: step_15_summary,
}
