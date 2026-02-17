"""Verification node — human-in-the-loop document confirmation via interrupt().

Uses event-driven priority routing: the router sends the user here
whenever verification_queue is non-empty, taking priority over
the next journey step.
"""

from langgraph.types import interrupt
from graph.state import LoanState
from workers import processing_store


def verification_node(state: LoanState) -> dict:
    """
    Pops ONE doc from verification_queue, presents extracted data,
    waits for user confirmation via interrupt().  Also syncs any
    newly completed documents from the processing store.
    """
    queue = list(state["verification_queue"])
    doc = queue[0]
    extracted = state["extracted_data"].get(doc, {})

    # Build a human-readable summary of extracted fields
    fields_display = "\n".join(f"  • **{k.replace('_', ' ').title()}**: `{v}`" for k, v in extracted.items())

    user_response = interrupt({
        "type": "document_verification",
        "document": doc,
        "extracted_fields": extracted,
        "message": (
            f"🔍 **{doc.replace('_', ' ').title()} — Verification**\n\n"
            f"{fields_display}\n\n"
            f"Reply **confirm** to accept, or provide corrections."
        ),
    })

    # Pop verified doc from queue, mark as verified
    new_queue = queue[1:]
    new_status = {**state["documents_status"], doc: "verified"}

    # Optionally merge corrections from user
    new_extracted = {**state["extracted_data"]}
    if isinstance(user_response, dict) and "corrections" in user_response:
        new_extracted[doc] = {**new_extracted.get(doc, {}), **user_response["corrections"]}

    result = {
        "verification_queue": new_queue,
        "documents_status": new_status,
        "extracted_data": new_extracted,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }

    # Sync any docs that finished processing while user was verifying
    completed = processing_store.pop_completed()
    if completed:
        for doc_type, data in completed:
            result["verification_queue"] = result["verification_queue"] + [doc_type]
            result["extracted_data"] = {**result["extracted_data"], doc_type: data}
            result["documents_status"] = {**result["documents_status"], doc_type: "ready_for_verification"}

    return result


def finish_node(state: LoanState) -> dict:
    """Terminal node — marks journey as complete."""
    return {"finished": True}
