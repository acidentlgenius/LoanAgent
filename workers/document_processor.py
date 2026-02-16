"""Async document processor — extracts structured data from uploaded documents.

In production this would be a Celery task or a background asyncio job.
For now, provides a simulated async interface.
"""

import asyncio
from typing import Any

# Expected fields per document type
DOC_FIELDS: dict[str, list[str]] = {
    "bank_statement": ["bank_name", "account_number", "ifsc", "account_holder", "balance"],
    "payslip": ["employer", "gross_salary", "net_salary", "pay_period"],
    "cibil": ["cibil_score", "report_date", "outstanding_loans"],
    "pan": ["pan_number", "name_on_pan", "dob"],
    "aadhaar": ["aadhaar_number", "name_on_aadhaar", "address"],
}


async def process_document(doc_type: str, file_path: str) -> dict[str, Any]:
    """
    Extract structured data from a document.
    In production: call an LLM with the extraction prompt.
    Here: returns placeholder extracted fields.
    """
    await asyncio.sleep(0.1)  # simulate I/O
    fields = DOC_FIELDS.get(doc_type, [])
    return {field: f"<extracted_{field}>" for field in fields}


def process_document_async(doc_type: str, file_path: str) -> None:
    """
    Fire-and-forget: schedule document processing in the background.
    The worker will update state (documents_status, extracted_data,
    verification_queue) via the checkpointer once extraction completes.
    """
    # In production: celery_app.send_task(...) or asyncio.create_task(...)
    # For development, this is a no-op placeholder — the test harness
    # simulates completed processing by directly updating state.
    pass
