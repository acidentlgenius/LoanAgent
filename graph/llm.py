"""LLM client — generates conversational prompts and extracts structured data.

The LLM is used ONLY for:
  ✅ Generating user-facing messages (contextual, warm, professional)
  ✅ Extracting structured data from free-text responses
  ✅ Validating / summarizing collected data
  ❌ NOT for routing or flow control (that's the router's job)
"""

import json
import logging
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field, create_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import GOOGLE_API_KEY, LLM_MODEL

# ── Step definitions (data-driven) ──────────────────────────────────────
STEP_DEFS: Dict[str, dict] = {
    "name": {
        "fields": {"first_name": "First name", "last_name": "Last name"},
        "ask": "the applicant's full name (first and last)",
        "example": "My name is Priya Sharma",
    },
    "dob": {
        "fields": {"date_of_birth": "Date of birth (YYYY-MM-DD)"},
        "ask": "their date of birth",
        "example": "I was born on 15 January 1990",
    },
    "contact": {
        "fields": {"phone": "Phone number", "email": "Email address"},
        "ask": "their phone number and email address",
        "example": "9876543210, priya@email.com",
    },
    "income": {
        "fields": {"monthly_income": "Monthly income amount", "income_source": "Source of income"},
        "ask": "their monthly income and its source",
        "example": "I earn 75,000 per month from my job as a software engineer",
    },
    "employment": {
        "fields": {"employer": "Employer name", "designation": "Job title", "tenure_years": "Years at current employer"},
        "ask": "their employment details — employer name, designation, and tenure",
        "example": "I work at TCS as a Senior Developer for 3 years",
    },
    "address": {
        "fields": {"full_address": "Complete current address", "city": "City", "pincode": "PIN code"},
        "ask": "their current residential address",
        "example": "42, MG Road, Bengaluru, Karnataka 560001",
    },
    "loan_amount": {
        "fields": {"amount": "Requested loan amount in INR"},
        "ask": "how much loan they are requesting",
        "example": "I need a loan of 5 lakhs",
    },
    "loan_tenure": {
        "fields": {"tenure_months": "Loan tenure in months"},
        "ask": "their preferred loan repayment tenure",
        "example": "I'd like to repay over 36 months",
    },
    "purpose": {
        "fields": {"loan_purpose": "Purpose of the loan"},
        "ask": "the purpose of this loan",
        "example": "I want to renovate my house",
    },
    "references": {
        "fields": {"ref_name": "Reference person's name", "ref_phone": "Reference phone", "ref_relation": "Relationship"},
        "ask": "a personal reference (name, phone, relationship)",
        "example": "Rahul Verma, 9988776655, colleague",
    },
    "bank_details": {
        "fields": {"account_number": "Bank account number", "ifsc": "IFSC code", "bank_name": "Bank name"},
        "ask": "their bank account details for loan disbursement",
        "example": "HDFC Bank, A/C 12345678, IFSC HDFC0001234",
    },
    "consent": {
        "fields": {"agreed": "Whether user agrees to T&C (yes/no)"},
        "ask": "whether they agree to the Terms & Conditions",
        "example": "Yes, I agree",
    },
    "document_upload": {
        "fields": {"documents": "List of documents being uploaded"},
        "ask": "to upload their documents (Bank Statement, Payslip, CIBIL, PAN, Aadhaar)",
        "example": "I'm uploading my bank statement and PAN card",
    },
}


# ── LLM instance (singleton) ───────────────────────────────────────────
_llm_instance = None


def _get_llm():
    """Lazy singleton — creates the LLM once, reuses on every call."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    if not GOOGLE_API_KEY:
        print("Wait, what? No API key found. Check your .env file.")
        return None
    _llm_instance = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )
    return _llm_instance


# ── Prompt cache (avoids repeated LLM call on interrupt replay) ────────
_prompt_cache: dict[tuple[str, int], str] = {}


# ── Prompt generation ──────────────────────────────────────────────────
async def generate_step_message(step_name: str, step_num: int, journey_data: dict) -> str:
    """Generate a warm, human-like prompt for a journey step (Async)."""
    cache_key = (step_name, step_num)
    if cache_key in _prompt_cache:
        return _prompt_cache[cache_key]

    step_def = STEP_DEFS.get(step_name, {})
    ask_desc = step_def.get("ask", step_name)

    # Build context from previous answers
    context_lines = []
    for key, val in journey_data.items():
        if isinstance(val, dict):
            readable = ", ".join(f"{k}: {v}" for k, v in val.items() if v)
            context_lines.append(f"  {key}: {readable}")
        else:
            context_lines.append(f"  {key}: {val}")
    context = "\n".join(context_lines) if context_lines else "  (Nothing yet)"

    llm = _get_llm()
    if not llm:
        return _template_prompt(step_name, step_num, journey_data)

    system = (
        "You are a warm, friendly loan advisor chatting with a customer. "
        "Talk like a real human — casual but professional, like a good bank relationship manager. "
        "RULES:\n"
        "- NEVER mention step numbers, field names, or form-like language\n"
        "- NEVER use bullet points or numbered lists\n"
        "- Acknowledge what the user just told you naturally (use their name if you know it)\n"
        "- Ask for the next piece of info in a conversational way\n"
        "- Keep it to 1-2 short sentences, like a text message\n"
        "- Vary your style — don't start every message the same way\n"
        "- Sound human, not like a bot reading a script"
    )
    human = (
        f"This is step {step_num} of 15 (don't mention this to the user).\n"
        f"You need to ask about: {ask_desc}\n"
        f"Data collected so far:\n{context}\n\n"
        f"Write your message to the customer."
    )

    try:
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        result = response.content.strip()
    except Exception as e:
        logging.error(f"Error generating prompt: {e}")
        result = _template_prompt(step_name, step_num, journey_data)

    _prompt_cache[cache_key] = result
    return result


def clear_prompt_cache():
    """Clear the prompt cache (used on session reset)."""
    _prompt_cache.clear()


def _template_prompt(step_name: str, step_num: int, journey_data: dict) -> str:
    """Fallback template when LLM is unavailable."""
    step_def = STEP_DEFS.get(step_name, {})
    ask = step_def.get("ask", step_name)
    name_data = journey_data.get("name", {})
    greeting = ""
    if isinstance(name_data, dict) and name_data.get("first_name"):
        greeting = f"Thanks, {name_data['first_name']}! "
    elif isinstance(name_data, str):
        greeting = f"Thanks! "
    return f"{greeting}Step {step_num}/15 — Please provide {ask}."


# ── Data extraction ────────────────────────────────────────────────────
async def extract_step_data(step_name: str, user_text: str) -> dict:
    """Extract structured fields from user's free-text response using LLM & Pydantic."""
    step_def = STEP_DEFS.get(step_name, {})
    fields = step_def.get("fields", {})

    llm = _get_llm()
    if not llm:
        return {"raw_input": user_text}

    # Dynamically create Pydantic model
    field_definitions = {k: (Optional[str], Field(None, description=desc)) for k, desc in fields.items()}
    DynamicModel = create_model(f"{step_name}_Extraction", **field_definitions)

    structured_llm = llm.with_structured_output(DynamicModel)

    system = (
        "You are a data extraction assistant. "
        "Extract the specified fields from the user's response. "
        "If a field is missing, leave it as null. "
        "Be smart about parsing — e.g., '5 lakhs' → '500000', 'three years' → '3'."
    )
    human = f"User's response: \"{user_text}\""

    try:
        extraction = await structured_llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        return extraction.model_dump(exclude_none=True)
    except Exception as e:
        logging.error(f"Error extracting data: {e}")
        return {"raw_input": user_text}


# ── Review / Summary generation ────────────────────────────────────────
async def generate_review_summary(journey_data: dict) -> str:
    """Generate a human-readable summary of all collected data for review."""
    llm = _get_llm()
    if not llm:
        return _fallback_summary(journey_data)

    data_json = json.dumps(journey_data, indent=2, default=str)
    system = (
        "You are a loan application assistant. "
        "Summarize the applicant's collected data in a clean, readable format. "
        "Use a friendly, professional tone. Organize by category. "
        "Use markdown formatting with bold labels."
    )
    human = f"Collected application data:\n{data_json}\n\nGenerate a review summary."

    try:
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        return response.content.strip()
    except Exception:
        return _fallback_summary(journey_data)


def _fallback_summary(journey_data: dict) -> str:
    lines = []
    for key, val in journey_data.items():
        if isinstance(val, dict):
            readable = ", ".join(f"{k}: {v}" for k, v in val.items())
            lines.append(f"• **{key.replace('_', ' ').title()}**: {readable}")
        else:
            lines.append(f"• **{key.replace('_', ' ').title()}**: {val}")
    return "Here's everything you've provided:\n\n" + "\n".join(lines)


async def generate_final_summary(journey_data: dict, documents_status: dict) -> str:
    """Generate the final application summary."""
    llm = _get_llm()
    if not llm:
        return "Your loan application has been submitted successfully! We'll contact you shortly."

    system = (
        "You are a loan application assistant. "
        "The applicant has completed all 15 steps. "
        "Generate a warm, professional final summary confirming submission. "
        "Mention key details (name, loan amount, tenure) if available. "
        "Keep it concise — 3-5 sentences."
    )
    human = (
        f"Application data: {json.dumps(journey_data, default=str)}\n"
        f"Document status: {json.dumps(documents_status)}\n"
        f"Generate the final confirmation message."
    )

    try:
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        return response.content.strip()
    except Exception:
        return "Your loan application has been submitted successfully! We'll be in touch soon."
