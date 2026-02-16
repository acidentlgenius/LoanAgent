"""Graph assembly — builds and compiles the LoanState graph with all nodes and edges."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import LoanState
from graph.router import router, STEP_NODE_MAP
from graph.journey_nodes import STEP_FUNCTIONS
from graph.verification_nodes import verification_node, finish_node


def build_graph(checkpointer=None):
    """
    Assemble the full loan journey graph.
    Returns a compiled graph ready for invoke/stream.
    """
    builder = StateGraph(LoanState)

    # ── Register journey step nodes ─────────────────────────────────
    for step_num, func in STEP_FUNCTIONS.items():
        builder.add_node(STEP_NODE_MAP[step_num], func)

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
