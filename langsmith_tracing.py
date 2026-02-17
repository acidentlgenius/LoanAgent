"""LangSmith tracing — one flow = one trace across interrupts."""

import os
import uuid
from contextlib import contextmanager

from config import LANGSMITH_TRACING, LANGSMITH_PROJECT

# In-memory store: thread_id -> parent RunTree (for REST API stateless resume)
_thread_trace_store: dict[str, "RunTree"] = {}


def _ensure_env():
    """Ensure LangSmith env vars are set when tracing is enabled."""
    if LANGSMITH_TRACING:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)


@contextmanager
def flow_trace(thread_id: str, user_id: str = ""):
    """
    Create a parent trace for a loan flow. All graph invokes inside this context
    are grouped under one trace. Use the same thread_id for start + all resumes.
    """
    if not LANGSMITH_TRACING:
        yield
        return

    _ensure_env()
    try:
        from langsmith.run_trees import RunTree
    except ImportError:
        yield
        return

    root = RunTree(
        name="loan_application_flow",
        run_type="chain",
    )
    root.add_metadata({"thread_id": thread_id, "user_id": user_id or thread_id})
    root.add_tags(["loan-agent", "flow"])
    root.post()
    _thread_trace_store[thread_id] = root

    try:
        import langsmith as ls

        with ls.tracing_context(
            project_name=LANGSMITH_PROJECT,
            enabled=True,
            parent=root,
            metadata={"thread_id": thread_id, "user_id": user_id or thread_id},
            tags=["loan-agent", "flow"],
        ):
            yield str(root.id)
    finally:
        pass  # Keep root in store for resume


@contextmanager
def continue_flow_trace(thread_id: str):
    """
    Continue an existing flow trace (for resume invokes). Use the parent run
    stored when the flow started.
    """
    if not LANGSMITH_TRACING:
        yield
        return

    _ensure_env()
    root = _thread_trace_store.get(thread_id)
    if not root:
        yield
        return

    try:
        import langsmith as ls
    except ImportError:
        yield
        return

    with ls.tracing_context(
        project_name=LANGSMITH_PROJECT,
        enabled=True,
        parent=root,
        metadata={"thread_id": thread_id},
        tags=["loan-agent", "flow", "resume"],
    ):
        yield


def clear_flow_trace(thread_id: str) -> None:
    """End the root run and remove from store when flow completes or resets."""
    root = _thread_trace_store.pop(thread_id, None)
    if root:
        try:
            root.end()
            root.patch()
        except Exception:
            pass
