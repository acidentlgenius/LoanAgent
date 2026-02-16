"""Verification node — human-in-the-loop document confirmation via interrupt()."""

from langgraph.types import interrupt
from graph.state import LoanState


def verification_node(state: LoanState) -> dict:
    """
    Pops ONE doc from verification_queue, presents extracted data,
    waits for user confirmation via interrupt().
    """
    queue = list(state["verification_queue"])
    doc = queue[0]
    extracted = state["extracted_data"].get(doc, {})

    # Pause for human review
    user_response = interrupt({
        "type": "document_verification",
        "document": doc,
        "extracted_fields": extracted,
        "message": f"Please verify the extracted data for your {doc.replace('_', ' ')}.",
    })

    # Pop verified doc from queue, mark as verified
    new_queue = queue[1:]
    new_status = {**state["documents_status"], doc: "verified"}

    # Optionally merge corrections from user
    new_extracted = {**state["extracted_data"]}
    if isinstance(user_response, dict) and "corrections" in user_response:
        new_extracted[doc] = {**new_extracted.get(doc, {}), **user_response["corrections"]}

    return {
        "verification_queue": new_queue,
        "documents_status": new_status,
        "extracted_data": new_extracted,
        "max_steps_guard": state["max_steps_guard"] + 1,
    }


def finish_node(state: LoanState) -> dict:
    """Terminal node — marks journey as complete."""
    return {"finished": True}
