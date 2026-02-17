"""Async document processor — simulates 2-10s extraction per document.

Spawns a background thread for each document.  When done, stores
dummy extracted data in the processing_store.  Journey nodes pick up
completed docs and inject them into verification_queue.
"""

import random
import time
import threading
from typing import Any

from workers import processing_store

# ── Dummy extracted data per document type ──────────────────────────────
DUMMY_DATA: dict[str, dict[str, Any]] = {
    "bank_statement": {
        "bank_name": "HDFC Bank",
        "ifsc": "HDFC0001234",
        "account_number": "50100012345678",
        "account_holder_name": "Abhinav Maurya",
    },
    "payslip": {
        "monthly_salary": "75000",
    },
    "cibil": {
        "cibil_score": "742",
    },
    "pan": {
        "pan_number": "ABCDE1234F",
    },
    "aadhaar": {
        "aadhaar_number": "1234 5678 9012",
    },
}


# Incremental delays so documents finish one-by-one during the journey
DOC_DELAYS: dict[str, float] = {
    "bank_statement": 2.0,
    "payslip": 10.0,
    "cibil": 20.0,
    "pan": 30.0,
    "aadhaar": 45.0,
}


def _process_worker(doc_type: str) -> None:
    """Background worker: sleep incremental seconds, then store dummy result."""
    delay = DOC_DELAYS.get(doc_type, 5.0)
    time.sleep(delay)
    data = DUMMY_DATA.get(doc_type, {"raw": f"<extracted_{doc_type}>"})
    processing_store.store_result(doc_type, data)


def process_document_async(doc_type: str, file_path: str) -> None:
    """
    Fire-and-forget: start a background thread to process the document.
    The thread will store results in the processing_store when done.
    """
    processing_store.mark_processing(doc_type)
    thread = threading.Thread(
        target=_process_worker,
        args=(doc_type,),
        daemon=True,
        name=f"doc-processor-{doc_type}",
    )
    thread.start()
