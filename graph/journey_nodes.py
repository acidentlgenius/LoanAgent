"""Journey nodes — Generic and specific steps for the loan application.

Refactored to shrink code and improve performance via async execution.
"""

import asyncio
from typing import Dict, Any

from langgraph.types import interrupt
from graph.state import LoanState
from graph.llm import (
    generate_step_message,
    extract_step_data,
    generate_review_summary,
    generate_final_summary,
    get_missing_required_fields,
    generate_missing_fields_prompt,
    validate_and_normalize,
)
from workers.document_processor import process_document_async
from workers import processing_store
from config import MAX_RETRIES_PER_STEP

# ── Document types expected at step 5 ──────────────────────────────────
REQUIRED_DOCUMENTS = ["bank_statement", "payslip", "cibil", "pan", "aadhaar"]


# ── Sync helper: inject completed docs into state ──────────────────────
def _sync_docs(state: LoanState) -> Dict[str, Any]:
    """
    Pop all completed docs from the processing_store and prepare state updates.
    """
    completed = processing_store.pop_completed()
    if not completed:
        return {}

    queue = list(state.get("verification_queue", []))
    extracted = dict(state.get("extracted_data", {}))
    doc_status = dict(state.get("documents_status", {}))

    for doc_type, data in completed:
        queue.append(doc_type)
        extracted[doc_type] = data
        doc_status[doc_type] = "ready_for_verification"

    return {
        "verification_queue": queue,
        "extracted_data": extracted,
        "documents_status": doc_status,
    }


# ── Universal Step Node ────────────────────────────────────────────────
async def universal_step_node(state: LoanState, step_name: str) -> Dict[str, Any]:
    """
    Generic LLM-powered step handler with retry for missing/invalid data.
    Used for all steps that follow the prompt -> interrupt -> extract pattern.
    Re-prompts (up to MAX_RETRIES_PER_STEP) if required fields are missing before advancing.
    """
    journey_data = dict(state["journey_data"])
    retries_left = MAX_RETRIES_PER_STEP
    prompt = await generate_step_message(step_name, state["current_step"], journey_data)

    while True:
        user_text = interrupt({
            "type": "journey_step",
            "step": state["current_step"],
            "field": step_name,
            "message": prompt,
        })

        # ── Runs only on resume (after interrupt returns) ──
        extracted = await extract_step_data(step_name, str(user_text))
        validated, _ = validate_and_normalize(step_name, extracted)
        merged = {**extracted, **validated}

        missing = get_missing_required_fields(step_name, merged)
        if not missing or retries_left <= 0:
            # Accept what we have and move on (or we exhausted retries)
            break

        # Re-prompt for missing fields
        retries_left -= 1
        prompt = await generate_missing_fields_prompt(step_name, missing, journey_data)

    result = {
        "journey_data": {**journey_data, step_name: merged},
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }

    # Inject any docs that finished processing while user was typing
    doc_updates = _sync_docs(state)
    result.update(doc_updates)

    return result


# ── Specific Nodes ──────────────────────────────────────────────────────

async def document_upload_node(state: LoanState) -> Dict[str, Any]:
    """Step 5: Collect document uploads — kicks off async processing."""
    prompt = await generate_step_message("document_upload", 5, state["journey_data"])

    # Simulate upload prompt
    interrupt({
        "type": "document_upload",
        "step": 5,
        "required_documents": REQUIRED_DOCUMENTS,
        "message": prompt,
    })

    # Accept any response — we start processing all 5 required docs
    new_uploaded = dict(state["documents_uploaded"])
    new_status = dict(state["documents_status"])

    for doc_type in REQUIRED_DOCUMENTS:
        new_uploaded[doc_type] = f"/uploads/{doc_type}.pdf"
        new_status[doc_type] = "processing"
        process_document_async(doc_type, f"/uploads/{doc_type}.pdf")

    return {
        "documents_uploaded": new_uploaded,
        "documents_status": new_status,
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


async def review_node(state: LoanState) -> Dict[str, Any]:
    """Step 13: Present LLM-generated summary for review.
    Async waits for all documents to be processed.
    """
    # Async non-blocking wait
    while processing_store.is_any_processing():
        await asyncio.sleep(0.5)

    # One final sync to pick up any remaining docs
    doc_updates = _sync_docs(state)

    summary = await generate_review_summary(state["journey_data"])

    user_input = interrupt({
        "type": "review",
        "step": 13,
        "collected_data": state["journey_data"],
        "documents_status": {**state["documents_status"], **doc_updates.get("documents_status", {})},
        "message": f"📋 **Application Review**\n\n{summary}\n\nPlease confirm everything looks correct, or let me know what to change.",
    })

    result = {
        "journey_data": {**state["journey_data"], "review_confirmed": str(user_input)},
        "current_step": state["current_step"] + 1,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }
    result.update(doc_updates)
    return result


async def summary_node(state: LoanState) -> Dict[str, Any]:
    """Step 14: Final summary — LLM generates confirmation."""
    final_msg = await generate_final_summary(state["journey_data"], state["documents_status"])

    interrupt({
        "type": "summary",
        "step": 14,
        "journey_data": state["journey_data"],
        "documents_status": state["documents_status"],
        "message": final_msg,
    })

    return {
        "current_step": 15,
        "max_steps_guard": state["max_steps_guard"] + 1,
        "finished": True,
    }
