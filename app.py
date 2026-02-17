"""Streamlit UI — chat-based interface for the LLM-powered loan journey."""

import streamlit as st
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from graph.state import initial_state
from graph.builder import build_graph
from workers import processing_store

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Agent", page_icon="🏦", layout="centered")

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    .doc-chip {
        display: inline-block; padding: 3px 10px; border-radius: 8px;
        margin: 2px 4px; font-size: 0.78em; font-weight: 500;
    }
    .doc-pending            { background: #FEF3C7; color: #92400E; }
    .doc-processing         { background: #DBEAFE; color: #1E40AF; }
    .doc-ready_for_verification { background: #FDE68A; color: #78350F; }
    .doc-verified           { background: #D1FAE5; color: #065F46; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ──────────────────────────────────────────────────
def _init_session():
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = "streamlit-main"
        st.session_state.config = {"configurable": {"thread_id": "streamlit-main"}}
        st.session_state.started = False
        st.session_state.messages = []
        st.session_state.pending_interrupt = None
        processing_store.reset()

_init_session()

graph = st.session_state.graph
config = st.session_state.config


# ── Helpers ─────────────────────────────────────────────────────────────
def get_interrupt():
    snapshot = graph.get_state(config)
    if snapshot and snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    return None


def get_state_values():
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot else {}


def format_interrupt(data: dict) -> str:
    """Format interrupt payload as a chat message."""
    return data.get("message", "")


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Loan Journey")
    vals = get_state_values()

    if vals:
        step = min(vals.get("current_step", 1), 15)
        st.progress((step - 1) / 15, text=f"Step {step} of 15")

        if vals.get("finished"):
            st.success("✅ Complete!")
        else:
            st.caption(f"Guard: {vals.get('max_steps_guard', 0)}/25")

        # Document status chips
        doc_status = vals.get("documents_status", {})
        if doc_status:
            st.markdown("**📄 Documents**")
            for doc, status in doc_status.items():
                emoji = {"processing": "⏳", "ready_for_verification": "🟡", "verified": "✅"}.get(status, "⬜")
                css_class = f"doc-{status}"
                st.markdown(
                    f'<span class="doc-chip {css_class}">'
                    f'{emoji} {doc.replace("_", " ").title()}: {status.replace("_", " ")}</span>',
                    unsafe_allow_html=True,
                )

            # Show processing indicator
            still_processing = processing_store.get_processing_docs()
            if still_processing:
                with st.status("Processing documents...", expanded=False):
                    for d in still_processing:
                        st.write(f"⏳ {d.replace('_', ' ').title()}")

        # Verification queue
        queue = vals.get("verification_queue", [])
        if queue:
            st.markdown("**🔍 Verification Queue**")
            for doc in queue:
                st.caption(f"  → {doc.replace('_', ' ').title()}")

        # Collected data preview
        journey = vals.get("journey_data", {})
        if journey:
            with st.expander("📝 Collected Data", expanded=False):
                for key, val in journey.items():
                    if isinstance(val, dict):
                        items = ", ".join(f"{k}: {v}" for k, v in val.items() if v)
                        st.caption(f"**{key}**: {items}")
                    else:
                        st.caption(f"**{key}**: {val}")

        # Extracted doc data
        extracted = vals.get("extracted_data", {})
        if extracted:
            with st.expander("📑 Extracted Data", expanded=False):
                for doc, fields in extracted.items():
                    st.caption(f"**{doc.replace('_', ' ').title()}**")
                    if isinstance(fields, dict):
                        for k, v in fields.items():
                            st.caption(f"  {k.replace('_', ' ').title()}: {v}")

    if st.button("🔄 Reset"):
        processing_store.reset()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ── Main ────────────────────────────────────────────────────────────────
st.title("🏦 Loan Application Agent")
st.caption("A conversational AI assistant guiding you through your loan application.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# Start button
if not st.session_state.started:
    if st.button("🚀 Start Loan Application", type="primary", use_container_width=True):
        graph.invoke(initial_state("streamlit-user"), config)
        st.session_state.started = True

        interrupt_data = get_interrupt()
        if interrupt_data:
            st.session_state.pending_interrupt = interrupt_data
            st.session_state.messages.append({
                "role": "assistant",
                "content": format_interrupt(interrupt_data),
                "avatar": "🏦",
            })
        st.rerun()

# Chat input
elif not get_state_values().get("finished", False):
    if user_text := st.chat_input("Type your response..."):
        st.session_state.messages.append({"role": "user", "content": user_text, "avatar": "👤"})

        # Parse input based on interrupt type
        interrupt_data = st.session_state.pending_interrupt or {}
        if interrupt_data.get("type") == "document_verification":
            if user_text.strip().lower() == "confirm":
                resume_data = {"confirmed": True}
            else:
                resume_data = user_text
        else:
            resume_data = user_text  # Free text → LLM extracts

        try:
            graph.invoke(Command(resume=resume_data), config)
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {e}",
                "avatar": "🏦",
            })
            st.rerun()

        next_interrupt = get_interrupt()
        if next_interrupt:
            st.session_state.pending_interrupt = next_interrupt
            st.session_state.messages.append({
                "role": "assistant",
                "content": format_interrupt(next_interrupt),
                "avatar": "🏦",
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "✅ Your loan application has been submitted!",
                "avatar": "🏦",
            })
        st.rerun()
else:
    st.balloons()
    st.success("🎉 Your loan application is complete!")
