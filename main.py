"""FastAPI entrypoint — exposes the LangGraph loan journey via REST."""

import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

from config import HOST, PORT
from graph.state import initial_state
from graph.builder import build_graph
from langsmith_tracing import flow_trace, continue_flow_trace, clear_flow_trace

# ── App + graph ─────────────────────────────────────────────────────────
app = FastAPI(title="Loan Agent", version="1.0.0")
graph = build_graph()


# ── Request / Response models ───────────────────────────────────────────
class StartRequest(BaseModel):
    user_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    data: dict | None = None  # user input passed via Command(resume=...)


# ── Endpoints ───────────────────────────────────────────────────────────

@app.post("/journey/start")
def start_journey(req: StartRequest):
    """Create a new loan application thread and run until first interrupt."""
    thread_id = str(uuid.uuid4())
    user_id = req.user_id or thread_id
    config = {"configurable": {"thread_id": thread_id}}

    with flow_trace(thread_id, user_id):
        result = graph.invoke(initial_state(user_id), config)

    return {
        "thread_id": thread_id,
        "state": result,
        "interrupt": _get_interrupt(config),
    }


@app.post("/journey/resume")
def resume_journey(req: ResumeRequest):
    """Resume the journey after an interrupt with user-provided data."""
    config = {"configurable": {"thread_id": req.thread_id}}

    # Check there is a pending interrupt
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.tasks:
        raise HTTPException(404, "No pending interrupt for this thread.")

    with continue_flow_trace(req.thread_id):
        result = graph.invoke(Command(resume=req.data), config)

    # Clear trace when flow is complete
    if result.get("finished"):
        clear_flow_trace(req.thread_id)

    return {
        "thread_id": req.thread_id,
        "state": result,
        "interrupt": _get_interrupt(config),
    }


@app.get("/journey/state/{thread_id}")
def get_journey_state(thread_id: str):
    """Retrieve current state for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(404, "Thread not found.")
    return {
        "thread_id": thread_id,
        "state": snapshot.values,
        "interrupt": _get_interrupt(config),
    }


# ── Helpers ─────────────────────────────────────────────────────────────
def _get_interrupt(config: dict) -> dict | None:
    """Extract the pending interrupt payload, if any."""
    snapshot = graph.get_state(config)
    if snapshot and snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    return None


# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
