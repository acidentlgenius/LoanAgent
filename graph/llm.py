"""LLM client — generates conversational prompts and extracts structured data.

The LLM is used ONLY for:
  ✅ Generating user-facing messages (contextual, warm, professional)
  ✅ Extracting structured data from free-text responses
  ✅ Validating / summarizing collected data
  ❌ NOT for routing or flow control (that's the router's job)
"""

import json
import logging
import re
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field, create_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import GOOGLE_API_KEY, LLM_MODEL

# ── Step definitions (data-driven) ──────────────────────────────────────
# Each step has "fields", "ask", "example", and "required" (list of required field keys)
STEP_DEFS: Dict[str, dict] = {
    "name": {
        "fields": {"first_name": "First name", "last_name": "Last name"},
        "required": ["first_name", "last_name"],
        "ask": "the applicant's full name (first and last)",
        "example": "My name is Priya Sharma",
    },
    "dob": {
        "fields": {"date_of_birth": "Date of birth (YYYY-MM-DD)"},
        "required": ["date_of_birth"],
        "ask": "their date of birth",
        "example": "I was born on 15 January 1990",
    },
    "contact": {
        "fields": {"phone": "Phone number (10 digits)", "email": "Valid email address"},
        "required": ["phone", "email"],
        "ask": "their phone number and email address",
        "example": "9876543210, priya@email.com",
    },
    "income": {
        "fields": {"monthly_income": "Monthly income amount", "income_source": "Source of income"},
        "required": ["monthly_income", "income_source"],
        "ask": "their monthly income and its source",
        "example": "I earn 75,000 per month from my job as a software engineer",
    },
    "employment": {
        "fields": {"employer": "Employer name", "designation": "Job title", "tenure_years": "Years at current employer"},
        "required": ["employer", "designation", "tenure_years"],
        "ask": "their employment details — employer name, designation, and tenure",
        "example": "I work at TCS as a Senior Developer for 3 years",
    },
    "address": {
        "fields": {"full_address": "Complete current address", "city": "City", "pincode": "PIN code (6 digits)"},
        "required": ["full_address", "city", "pincode"],
        "ask": "their current residential address",
        "example": "42, MG Road, Bengaluru, Karnataka 560001",
    },
    "loan_amount": {
        "fields": {"amount": "Requested loan amount in INR (numeric)"},
        "required": ["amount"],
        "ask": "how much loan they are requesting",
        "example": "I need a loan of 5 lakhs",
    },
    "loan_tenure": {
        "fields": {"tenure_months": "Loan tenure in months"},
        "required": ["tenure_months"],
        "ask": "their preferred loan repayment tenure",
        "example": "I'd like to repay over 36 months",
    },
    "purpose": {
        "fields": {"loan_purpose": "Purpose of the loan"},
        "required": ["loan_purpose"],
        "ask": "the purpose of this loan",
        "example": "I want to renovate my house",
    },
    "references": {
        "fields": {"ref_name": "Reference person's name", "ref_phone": "Reference phone (10 digits)", "ref_relation": "Relationship"},
        "required": ["ref_name", "ref_phone", "ref_relation"],
        "ask": "a personal reference (name, phone, relationship)",
        "example": "Rahul Verma, 9988776655, colleague",
    },
    "bank_details": {
        "fields": {"account_number": "Bank account number", "ifsc": "IFSC code (e.g. HDFC0001234)", "bank_name": "Bank name"},
        "required": ["account_number", "ifsc", "bank_name"],
        "ask": "their bank account details for loan disbursement",
        "example": "HDFC Bank, A/C 12345678, IFSC HDFC0001234",
    },
    "consent": {
        "fields": {"agreed": "Whether user agrees to T&C (yes/no)"},
        "required": ["agreed"],
        "ask": "whether they agree to the Terms & Conditions",
        "example": "Yes, I agree",
    },
    "document_upload": {
        "fields": {"documents": "List of documents being uploaded"},
        "required": [],
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
        # Use astream to emit on_chat_model_stream events for the graph
        full_response = ""
        async for chunk in llm.astream([SystemMessage(content=system), HumanMessage(content=human)]):
            full_response += chunk.content
        result = full_response.strip()
    except Exception as e:
        logging.error(f"Error generating prompt: {e}")
        result = _template_prompt(step_name, step_num, journey_data)

    _prompt_cache[cache_key] = result
    return result


def clear_prompt_cache():
    """Clear the prompt cache (used on session reset)."""
    _prompt_cache.clear()


def clear_llm_instance():
    """
    Reset the LLM singleton. Use before each asyncio.run() in Streamlit to avoid
    'Event loop is closed' errors — the cached LLM holds HTTP clients tied to a
    previous event loop that gets closed between reruns.
    """
    global _llm_instance
    _llm_instance = None


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


# ── Field validators (Pydantic-style) ────────────────────────────────────
def _normalize_phone(val: str) -> str | None:
    """Extract 10-digit Indian phone number."""
    if not val:
        return None
    digits = re.sub(r"\D", "", str(val))
    return digits[-10:] if len(digits) >= 10 else None


def _normalize_email(val: str) -> str | None:
    """Basic email format check."""
    if not val:
        return None
    val = str(val).strip().lower()
    if re.match(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$", val):
        return val
    return None


def _normalize_date(val: str) -> str | None:
    """Expect YYYY-MM-DD or parse common formats."""
    if not val:
        return None
    val = str(val).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        return val
    # Try to extract YYYY-MM-DD from "15 January 1990" style
    months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05",
              "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
    m = re.search(r"(\d{1,2})[\s\-/]*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-/]*(\d{4})", val, re.I)
    if m:
        d, mo, y = m.group(1).zfill(2), months.get(m.group(2).lower()[:3], "01"), m.group(3)
        return f"{y}-{mo}-{d}"
    return None


def _normalize_amount(val: str) -> str | None:
    """Extract numeric amount; '5 lakhs' → '500000', cap at 50L for personal loans."""
    if not val:
        return None
    val = str(val).lower().strip()
    val = re.sub(r"[^\d.lakhs?lakhlac]", "", val)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lakh|lac)?", val)
    if not m:
        return None
    num = float(m.group(1))
    if "lakh" in val or "lac" in val:
        num *= 100_000
    amount = str(int(num))
    if int(amount) > 50_000_000:  # 50L sanity cap
        return None  # Flag for confirmation
    return amount


def _normalize_pincode(val: str) -> str | None:
    """6-digit Indian PIN code."""
    if not val:
        return None
    digits = re.sub(r"\D", "", str(val))
    return digits[:6] if len(digits) >= 6 else None


def _normalize_ifsc(val: str) -> str | None:
    """IFSC: 4 letters + 7 alphanumeric."""
    if not val:
        return None
    val = str(val).strip().upper()
    m = re.search(r"([A-Z]{4}0[A-Z0-9]{6})", val)
    return m.group(1) if m else None


# Map step+field → normalizer
FIELD_NORMALIZERS: Dict[tuple[str, str], Any] = {
    ("contact", "phone"): _normalize_phone,
    ("contact", "email"): _normalize_email,
    ("dob", "date_of_birth"): _normalize_date,
    ("address", "pincode"): _normalize_pincode,
    ("loan_amount", "amount"): _normalize_amount,
    ("references", "ref_phone"): _normalize_phone,
    ("bank_details", "ifsc"): _normalize_ifsc,
}


def validate_and_normalize(step_name: str, extracted: dict) -> tuple[dict, List[str]]:
    """
    Apply Pydantic-style validators and normalizers.
    Returns (validated_dict, list_of_validation_errors).
    """
    validated = {}
    errors = []
    step_def = STEP_DEFS.get(step_name, {})
    fields = step_def.get("fields", {})

    for key, raw in extracted.items():
        if key not in fields:
            continue
        normalizer = FIELD_NORMALIZERS.get((step_name, key))
        if normalizer:
            try:
                out = normalizer(raw)
                if out is None:
                    errors.append(f"{key}: invalid or missing value")
                else:
                    validated[key] = str(out)
            except Exception:
                errors.append(f"{key}: could not parse")
        else:
            if raw and str(raw).strip():
                validated[key] = str(raw).strip()
            elif key in step_def.get("required", []):
                errors.append(f"{key}: required")

    # Add non-validated fields that have values
    for k, v in extracted.items():
        if k not in validated and v and str(v).strip() and k in fields:
            validated[k] = str(v).strip()

    return validated, errors


def get_missing_required_fields(step_name: str, extracted: dict) -> List[str]:
    """Return list of required field names that are missing or invalid."""
    step_def = STEP_DEFS.get(step_name, {})
    required = step_def.get("required", [])
    _, errors = validate_and_normalize(step_name, extracted)
    missing = []
    for r in required:
        val = extracted.get(r)
        if not val or not str(val).strip():
            missing.append(r)
        elif (step_name, r) in FIELD_NORMALIZERS:
            norm = FIELD_NORMALIZERS[(step_name, r)](val)
            if norm is None:
                missing.append(r)
    return missing


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
        "Be smart about parsing — e.g., '5 lakhs' → '500000', '15 Jan 1990' → '1990-01-15', "
        "'9876543210' from 'call me on 98-7654-3210'. Normalize phone to 10 digits, email to lowercase."
    )
    human = f"User's response: \"{user_text}\""

    try:
        extraction = await structured_llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = extraction.model_dump(exclude_none=True)
        validated, _ = validate_and_normalize(step_name, raw)
        return validated if validated else raw
    except Exception as e:
        logging.error(f"Error extracting data: {e}")
        return {"raw_input": user_text}


async def generate_missing_fields_prompt(step_name: str, missing_fields: List[str], journey_data: dict) -> str:
    """Generate a friendly prompt asking for the missing/invalid fields."""
    step_def = STEP_DEFS.get(step_name, {})
    fields = step_def.get("fields", {})
    labels = [fields.get(f, f.replace("_", " ")) for f in missing_fields]

    llm = _get_llm()
    if not llm:
        return f"Could you please provide: {', '.join(labels)}?"

    system = (
        "You are a warm loan advisor. The applicant missed some details. "
        "Politely ask for ONLY the missing items in one short sentence. Don't repeat everything."
    )
    human = (
        f"Missing/invalid fields: {', '.join(labels)}\n"
        f"Already collected: {json.dumps(journey_data, default=str)}\n"
        f"Write a single friendly follow-up asking for these specific items."
    )

    try:
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        return response.content.strip()
    except Exception:
        return f"I still need: {', '.join(labels)}. Could you provide those?"


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
