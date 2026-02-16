"""Streamlit UI — chat-based interface for the 15-step loan journey."""

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
    .step-badge {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        background: #4F46E5; color: white; font-size: 0.8em; font-weight: 600;
    }
    .doc-chip {
        display: inline-block; padding: 2px 8px; border-radius: 8px;
        margin: 2px; font-size: 0.75em; font-weight: 500;
    }
    .doc-pending   { background: #FEF3C7; color: #92400E; }
    .doc-processing{ background: #DBEAFE; color: #1E40AF; }
    .doc-completed { background: #D1FAE5; color: #065F46; }
    .doc-verified  { background: #E0E7FF; color: #3730A3; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ──────────────────────────────────────────────────
def _init_session():
    """Initialize session state on first load."""
    if "graph" not in st.session_state:
        checkpointer = MemorySaver()
        st.session_state.graph = build_graph(checkpointer=checkpointer)
        st.session_state.thread_id = "streamlit-main"
        st.session_state.config = {"configurable": {"thread_id": "streamlit-main"}}
        st.session_state.started = False
        st.session_state.messages = []  # chat history
        st.session_state.pending_interrupt = None

_init_session()

graph = st.session_state.graph
config = st.session_state.config


# ── Helpers ─────────────────────────────────────────────────────────────
def get_interrupt():
    """Extract the current interrupt payload from graph state."""
    snapshot = graph.get_state(config)
    if snapshot and snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    return None


def get_state_values():
    """Get current graph state values."""
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot else {}


def format_interrupt_message(interrupt_data: dict) -> str:
    """Format an interrupt payload into a user-friendly message."""
    msg_type = interrupt_data.get("type", "")
    message = interrupt_data.get("message", "")

    if msg_type == "document_upload":
        docs = interrupt_data.get("required_documents", [])
        doc_list = "\n".join(f"  • {d.replace('_', ' ').title()}" for d in docs)
        return f"📄 **{message}**\n\n{doc_list}"

    if msg_type == "document_verification":
        doc = interrupt_data.get("document", "")
        fields = interrupt_data.get("extracted_fields", {})
        field_lines = "\n".join(f"  • **{k}**: {v}" for k, v in fields.items())
        return f"🔍 **{message}**\n\n{field_lines}\n\nType `confirm` to verify, or provide corrections as JSON."

    if msg_type == "review":
        data = interrupt_data.get("collected_data", {})
        lines = "\n".join(f"  • **{k}**: {v}" for k, v in data.items())
        return f"📋 **{message}**\n\n{lines}"

    if msg_type == "summary":
        return f"✅ **{message}**"

    return f"💬 **{message}**"


def parse_user_input(text: str, interrupt_data: dict) -> any:
    """Parse user text input based on the current interrupt type."""
    msg_type = interrupt_data.get("type", "")

    # Document upload: expect comma-separated doc_type:path pairs
    if msg_type == "document_upload":
        pairs = {}
        for part in text.split(","):
            part = part.strip()
            if ":" in part:
                doc_type, path = part.split(":", 1)
                pairs[doc_type.strip().lower().replace(" ", "_")] = path.strip()
        return pairs if pairs else text

    # Verification: "confirm" or JSON corrections
    if msg_type == "document_verification":
        if text.strip().lower() == "confirm":
            return {"confirmed": True}
        try:
            import json
            return {"corrections": json.loads(text)}
        except Exception:
            return {"confirmed": True}

    # Default: return raw text (or try dict-like)
    return text


# ── Sidebar: state dashboard ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Loan Journey")
    state_vals = get_state_values()

    if state_vals:
        step = state_vals.get("current_step", 1)
        guard = state_vals.get("max_steps_guard", 0)
        finished = state_vals.get("finished", False)

        # Progress bar
        progress = min((step - 1) / 15, 1.0)
        st.progress(progress, text=f"Step {min(step, 15)} of 15")

        if finished:
            st.success("✅ Journey Complete!")
        else:
            st.info(f"Guard counter: {guard}/25")

        # Document status chips
        doc_status = state_vals.get("documents_status", {})
        if doc_status:
            st.markdown("#### 📄 Documents")
            for doc, status in doc_status.items():
                css_class = f"doc-{status}"
                st.markdown(
                    f'<span class="doc-chip {css_class}">{doc.replace("_", " ").title()}: {status}</span>',
                    unsafe_allow_html=True,
                )

        # Verification queue
        queue = state_vals.get("verification_queue", [])
        if queue:
            st.markdown("#### 🔍 Verification Queue")
            for doc in queue:
                st.markdown(f"  • {doc.replace('_', ' ').title()}")

    if st.button("🔄 Reset Journey"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ── Main chat area ──────────────────────────────────────────────────────
st.title("🏦 Loan Application Agent")
st.caption("Walk through 15 steps to complete your loan application.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Start journey ───────────────────────────────────────────────────────
if not st.session_state.started:
    if st.button("🚀 Start Loan Application", type="primary", use_container_width=True):
        # Invoke graph with initial state
        graph.invoke(initial_state("streamlit-user"), config)
        st.session_state.started = True

        # Get first interrupt
        interrupt_data = get_interrupt()
        if interrupt_data:
            st.session_state.pending_interrupt = interrupt_data
            bot_msg = format_interrupt_message(interrupt_data)
            st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        st.rerun()

# ── Chat input ──────────────────────────────────────────────────────────
elif not get_state_values().get("finished", False):
    if user_text := st.chat_input("Your response..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Parse input based on pending interrupt type
        interrupt_data = st.session_state.pending_interrupt or {}
        parsed = parse_user_input(user_text, interrupt_data)

        # Resume the graph
        try:
            graph.invoke(Command(resume=parsed), config)
        except Exception as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {e}"}
            )
            st.rerun()

        # Check for next interrupt
        next_interrupt = get_interrupt()
        if next_interrupt:
            st.session_state.pending_interrupt = next_interrupt
            bot_msg = format_interrupt_message(next_interrupt)
            st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        else:
            # Journey complete
            st.session_state.messages.append(
                {"role": "assistant", "content": "✅ **Your loan application has been submitted!**"}
            )

        st.rerun()
else:
    st.balloons()
    st.success("🎉 Your loan application journey is complete! Thank you.")
