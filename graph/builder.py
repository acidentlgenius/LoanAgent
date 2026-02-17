"""Graph assembly — builds and compiles the LoanState graph with all nodes and edges."""

from functools import partial

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import LoanState
from graph.router import router, STEP_NODE_MAP
from graph.journey_nodes import (
    universal_step_node,
    document_upload_node,
    review_node,
    summary_node,
)
from graph.verification_nodes import verification_node, finish_node


def build_graph(checkpointer=None):
    """
    Assemble the full loan journey graph.
    Returns a compiled graph ready for invoke/stream.
    """
    builder = StateGraph(LoanState)

    # ── Register journey step nodes ─────────────────────────────────
    # We dynamically register nodes based on the router's map
    for step_num, node_name in STEP_NODE_MAP.items():
        if step_num == 5:
            builder.add_node(node_name, document_upload_node)
        elif step_num == 14:
            builder.add_node(node_name, review_node)
        elif step_num == 15:
            builder.add_node(node_name, summary_node)
        else:
            # Extract step name from node_name (e.g., "step_1_name" -> "name")
            # Format is always: step_N_suffix
            parts = node_name.split("_", 2)
            if len(parts) < 3:
                # Fallback or error, but we know the map is consistent
                continue
            step_suffix = parts[2]
            
            # Create a partial function for this specific step
            node_func = partial(universal_step_node, step_name=step_suffix)
            # Partial functions don't have __name__, which LangGraph might use for default naming, 
            # but we explicitly provide node_name in add_node, so it should be fine.
            # Just in case, we can attach a name.
            node_func.__name__ = node_name
            
            builder.add_node(node_name, node_func)

    # ── Register verification + finish nodes ────────────────────────
    builder.add_node("verification_node", verification_node)
    builder.add_node("finish", finish_node)

    # ── Entry edge ──────────────────────────────────────────────────
    builder.add_edge(START, "step_1_name")

    # ── Conditional edges: every step → router ──────────────────────
    for node_name in STEP_NODE_MAP.values():
        builder.add_conditional_edges(node_name, router)

    # ── Verification → router (may go back to journey or more verification)
    builder.add_conditional_edges("verification_node", router)

    # ── Finish → END ────────────────────────────────────────────────
    builder.add_edge("finish", END)

    # ── Compile with checkpointer (required for interrupt) ──────────
    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
