"""Streamlit UI — chat-based interface for the LLM-powered loan journey."""

import streamlit as st
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from graph.state import initial_state
from graph.builder import build_graph

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
    .doc-pending    { background: #FEF3C7; color: #92400E; }
    .doc-processing { background: #DBEAFE; color: #1E40AF; }
    .doc-completed  { background: #D1FAE5; color: #065F46; }
    .doc-verified   { background: #E0E7FF; color: #3730A3; }
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

_init_session()

graph = st.session_state.graph
config = st.session_state.config


# ── Helpers ─────────────────────────────────────────────────────────────
def get_interrupt():
    """Extract the pending interrupt payload."""
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
    msg_type = data.get("type", "")
    message = data.get("message", "")

    if msg_type == "document_verification":
        fields = data.get("extracted_fields", {})
        field_lines = "\n".join(f"  • **{k}**: `{v}`" for k, v in fields.items())
        return f"{message}\n\n{field_lines}\n\nReply **confirm** or provide corrections."

    # All other types: the LLM already generated a natural message
    return message


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

        # Document chips
        doc_status = vals.get("documents_status", {})
        if doc_status:
            st.markdown("**📄 Documents**")
            for doc, status in doc_status.items():
                st.markdown(
                    f'<span class="doc-chip doc-{status}">'
                    f'{doc.replace("_", " ").title()}: {status}</span>',
                    unsafe_allow_html=True,
                )

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

    if st.button("🔄 Reset"):
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
            bot_msg = format_interrupt(interrupt_data)
            st.session_state.messages.append({"role": "assistant", "content": bot_msg, "avatar": "🏦"})
        st.rerun()

# Chat input
elif not get_state_values().get("finished", False):
    if user_text := st.chat_input("Type your response..."):
        st.session_state.messages.append({"role": "user", "content": user_text, "avatar": "👤"})

        # Parse document uploads specially
        interrupt_data = st.session_state.pending_interrupt or {}
        if interrupt_data.get("type") == "document_upload" and ":" in user_text:
            parsed = {}
            for part in user_text.split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    parsed[k.strip().lower().replace(" ", "_")] = v.strip()
            resume_data = parsed or user_text
        elif interrupt_data.get("type") == "document_verification":
            resume_data = {"confirmed": True} if user_text.strip().lower() == "confirm" else user_text
        else:
            resume_data = user_text  # Free text → LLM will extract

        try:
            graph.invoke(Command(resume=resume_data), config)
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}", "avatar": "🏦"})
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
