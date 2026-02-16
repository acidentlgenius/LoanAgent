"""LLM prompt templates for document data extraction.

Used by the document processor to extract structured fields.
In production, these are sent to the LLM along with the document content.
"""

EXTRACTION_PROMPTS: dict[str, str] = {
    "bank_statement": (
        "Extract the following fields from this bank statement:\n"
        "- bank_name\n- account_number\n- ifsc\n- account_holder\n- balance\n"
        "Return as JSON."
    ),
    "payslip": (
        "Extract the following fields from this payslip:\n"
        "- employer\n- gross_salary\n- net_salary\n- pay_period\n"
        "Return as JSON."
    ),
    "cibil": (
        "Extract the following fields from this CIBIL report:\n"
        "- cibil_score\n- report_date\n- outstanding_loans\n"
        "Return as JSON."
    ),
    "pan": (
        "Extract the following fields from this PAN card:\n"
        "- pan_number\n- name_on_pan\n- dob\n"
        "Return as JSON."
    ),
    "aadhaar": (
        "Extract the following fields from this Aadhaar card:\n"
        "- aadhaar_number\n- name_on_aadhaar\n- address\n"
        "Return as JSON."
    ),
}


def get_extraction_prompt(doc_type: str, document_text: str) -> str:
    """Build a full extraction prompt for the given document type."""
    base = EXTRACTION_PROMPTS.get(doc_type, "Extract all relevant fields. Return as JSON.")
    return f"{base}\n\n---\nDocument content:\n{document_text}"
