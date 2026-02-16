"""15 journey step nodes — each collects data via interrupt() and advances state."""

from langgraph.types import interrupt
from graph.state import LoanState
from workers.document_processor import process_document_async

# ── Document types expected at step 5 ──────────────────────────────────
REQUIRED_DOCUMENTS = ["bank_statement", "payslip", "cibil", "pan", "aadhaar"]


# ── Helper ──────────────────────────────────────────────────────────────
def _base_step(state: LoanState, step_name: str, prompt: str) -> dict:
    """Generic step: interrupt for input, store it, advance."""
    user_input = interrupt({
        "type": "journey_step",
        "step": state["current_step"],
        "field": step_name,
        "message": prompt,
    })
    journey = {**state["journey_data"], step_name: user_input}
    return {
        "journey_data": journey,
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


# ── Step functions ──────────────────────────────────────────────────────

def step_1_name(state: LoanState) -> dict:
    """Collect first name and last name."""
    return _base_step(state, "name", "Please provide your first name and last name.")


def step_2_dob(state: LoanState) -> dict:
    """Collect date of birth."""
    return _base_step(state, "dob", "Please provide your date of birth (YYYY-MM-DD).")


def step_3_contact(state: LoanState) -> dict:
    """Collect phone and email."""
    return _base_step(state, "contact", "Please provide your phone number and email address.")


def step_4_income(state: LoanState) -> dict:
    """Collect monthly income and source."""
    return _base_step(state, "income", "Please provide your monthly income and its source.")


def step_5_document_upload(state: LoanState) -> dict:
    """Collect document uploads — kicks off async processing for each."""
    user_input = interrupt({
        "type": "document_upload",
        "step": 5,
        "required_documents": REQUIRED_DOCUMENTS,
        "message": "Please upload: Bank Statement, Payslip, CIBIL, PAN, Aadhaar.",
    })

    # user_input expected: {doc_type: file_path, …}
    uploads = user_input if isinstance(user_input, dict) else {}
    new_uploaded = {**state["documents_uploaded"], **uploads}
    new_status = {**state["documents_status"]}

    # Mark each uploaded doc as "processing" and fire async worker
    for doc_type, file_path in uploads.items():
        new_status[doc_type] = "processing"
        process_document_async(doc_type, file_path)  # non-blocking

    return {
        "documents_uploaded": new_uploaded,
        "documents_status": new_status,
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


def step_6_employment(state: LoanState) -> dict:
    """Collect employer, designation, tenure."""
    return _base_step(state, "employment", "Please provide employer name, designation, and tenure.")


def step_7_address(state: LoanState) -> dict:
    """Collect current address."""
    return _base_step(state, "address", "Please provide your current address.")


def step_8_loan_amount(state: LoanState) -> dict:
    """Collect requested loan amount."""
    return _base_step(state, "loan_amount", "What loan amount are you requesting?")


def step_9_loan_tenure(state: LoanState) -> dict:
    """Collect loan tenure in months."""
    return _base_step(state, "loan_tenure", "What is your preferred loan tenure (in months)?")


def step_10_purpose(state: LoanState) -> dict:
    """Collect loan purpose."""
    return _base_step(state, "purpose", "What is the purpose of this loan?")


def step_11_references(state: LoanState) -> dict:
    """Collect references / guarantor info."""
    return _base_step(state, "references", "Please provide reference/guarantor details.")


def step_12_bank_details(state: LoanState) -> dict:
    """Collect disbursement bank account."""
    return _base_step(state, "bank_details", "Please provide your bank account details for disbursement.")


def step_13_consent(state: LoanState) -> dict:
    """Collect T&C consent."""
    return _base_step(state, "consent", "Do you agree to the Terms & Conditions? (yes/no)")


def step_14_review(state: LoanState) -> dict:
    """Present all collected data for review."""
    user_input = interrupt({
        "type": "review",
        "step": 14,
        "collected_data": state["journey_data"],
        "documents_status": state["documents_status"],
        "message": "Please review all your submitted information. Confirm to proceed.",
    })
    journey = {**state["journey_data"], "review_confirmed": user_input}
    return {
        "journey_data": journey,
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


def step_15_summary(state: LoanState) -> dict:
    """Final summary — mark journey finished."""
    interrupt({
        "type": "summary",
        "step": 15,
        "journey_data": state["journey_data"],
        "documents_status": state["documents_status"],
        "message": "Your loan application has been submitted. Here is your summary.",
    })
    return {
        "current_step": 16,  # past last step → router will route to finish
        "max_steps_guard": state["max_steps_guard"] + 1,
        "finished": True,
    }


# Registry: step number → function (used by builder)
STEP_FUNCTIONS: dict[int, callable] = {
    1: step_1_name,
    2: step_2_dob,
    3: step_3_contact,
    4: step_4_income,
    5: step_5_document_upload,
    6: step_6_employment,
    7: step_7_address,
    8: step_8_loan_amount,
    9: step_9_loan_tenure,
    10: step_10_purpose,
    11: step_11_references,
    12: step_12_bank_details,
    13: step_13_consent,
    14: step_14_review,
    15: step_15_summary,
}
