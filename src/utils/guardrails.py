import re
from langchain_ollama import ChatOllama
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

REJECT_PATTERNS = re.compile(
    r'<[^>]+'
    r'|javascript\s*:'
    r'|on\w+\s*='
    r'|ignore\s+(previous|all|above)\s+instructions'
    r'|forget\s+(previous|all|above)\s+instructions'
    r'|system\s*prompt'
    r'|union\s+select|drop\s+table'
    r'|;\s*(drop|delete|update|insert)',
    re.IGNORECASE
)

UNSAFE_KEYWORDS = re.compile(
    r'\b(password|credit.?card|ssn|social.?security)\b',
    re.IGNORECASE
)

_LOOKS_CLEAN = re.compile(r"^[a-zA-Z0-9\s\?\.\,\'\"!\-]+$")

_COMMON_WORDS = {
    "what", "where", "when", "who", "why", "how",
    "is", "are", "was", "were", "do", "does", "can", "tell",
    "give", "show", "explain", "list", "describe",
    "the", "a", "an", "me", "my", "by", "to", "of", "about",
    "your", "our", "their", "its",
    "company", "mission", "vision", "product", "products", "pricing",
    "price", "shipping", "return", "returns", "policy", "policies",
    "contact", "support", "timings", "hours", "working", "details",
    "information", "services", "plans", "features", "founded", "location",
    "address", "email", "phone", "refund", "delivery", "charges",
    "mail", "id", "number", "name", "get", "find", "know", "need",
    "want", "have"
}

_SAFE_SHORT = {
    "is", "a", "an", "me", "do", "my", "by", "to", "of",
    "the", "are", "how", "why", "who", "tell", "was", "its",
    "our", "can"
}


async def rewrite_query(text: str, llm: ChatOllama) -> tuple[str, bool]:
    """
    Use LLM to fix spelling and grammar mistakes in the user query.
    Clean queries are detected and skipped — no LLM call needed.
    Falls back to original query if LLM call fails or returns garbage.
    """
    words = text.strip().split()
    clean_words = [w.strip("?.,!\"'").lower() for w in words]

    is_clean = bool(_LOOKS_CLEAN.match(text))
    known = sum(1 for w in clean_words if w in _COMMON_WORDS)
    known_ratio = known / len(clean_words) if clean_words else 0
    has_bad_short = any(
        len(w) <= 3 and w not in _SAFE_SHORT
        for w in clean_words
    )

    if is_clean and known_ratio >= 0.6 and not has_bad_short:
        logger.info(f"Clean query — LLM skipped (known_ratio={known_ratio:.2f})")
        return text, False

    logger.info(f"Rewriting query (known_ratio={known_ratio:.2f}, has_bad_short={has_bad_short})")

    prompt = f"""You are a query corrector for a company information chatbot.
Users ask questions about the company's mission, vision, products, pricing,
shipping, returns, contact details, and policies.

IMPORTANT RULES:
1. If the question is already correct and clear, return it EXACTLY as is — no changes at all.
2. Only fix if there are CLEAR spelling mistakes.
3. Do not rephrase, restructure, or add words.
4. Return ONLY the question, nothing else. No explanation.

Examples of correct queries — return EXACTLY as is:
- "Tell me about the company" -> "Tell me about the company"
- "What are the company support timings?" -> "What are the company support timings?"
- "What is the mission?" -> "What is the mission?"
- "What are your products?" -> "What are your products?"

Examples that need fixing:
- "wht is the mision" -> "what is the mission"
- "compny polcy" -> "company policy"
- "shpping rtrn" -> "shipping return"
- "contct detils" -> "contact details"
- "pric of prodct" -> "price of product"
- "wht area the compsh misisokmn" -> "what are the company mission"
- "tel me abot the compy" -> "tell me about the company"
- "wht r ur rtrn plcy" -> "what are your return policy"
- "hw do i contct suport" -> "how do i contact support"
- "wht ar the wrking hrs" -> "what are the working hours"

Question: {text}
Corrected:"""

    try:
        result = await llm.ainvoke(prompt)
        corrected = result.content.strip()

        if not corrected or len(corrected) > len(text) * 3:
            logger.warning("LLM rewrite returned unexpected result — using original")
            return text, False

        corrected = corrected.splitlines()[0].strip()
        was_corrected = corrected.strip().lower() != text.strip().lower()

        if was_corrected:
            logger.info(f"LLM query rewrite: '{text}' to '{corrected}'")
        else:
            logger.info("Query already correct — no rewrite needed")

        return corrected, was_corrected

    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return text, False


def validate_input(text: str) -> dict:
    """
    Validate user input — reject if dangerous pattern found.
    Spelling correction happens separately via rewrite_query() in ask.py.
    """
    if not text.strip():
        logger.warning("Empty input provided")
        raise ValueError("Please enter a valid question.")

    if len(text) > 500:
        logger.warning(f"Input too long: {len(text)} chars")
        raise ValueError("Your question is too long. Please keep it under 500 characters.")

    if REJECT_PATTERNS.search(text):
        logger.warning(f"Dangerous pattern detected in input: '{text[:50]}'")
        raise ValueError("Please enter a valid question without special characters or code.")

    logger.info("Input validation passed")

    return {
        "query": text.strip(),
        "original_query": text.strip(),
        "was_corrected": False,
        "correction_note": None,
    }


async def inline_model_guardrail(answer: str, llm: ChatOllama) -> str:
    """
    Async safety check on the LLM-generated answer.
    Step 1 — Fast regex keyword check (no LLM call).
    Step 2 — Full LLM safety check (only if step 1 passes).
    Returns original answer if SAFE, fallback message if UNSAFE.
    Raises RuntimeError if the LLM call fails.
    """
    if not answer.strip():
        logger.warning("Empty answer provided to guardrail")
        return "No answer was generated."

    if UNSAFE_KEYWORDS.search(answer):
        logger.warning("Unsafe keyword detected in answer — blocked without LLM call")
        return "I cannot share this information."

    safety_prompt = f"""Reply with SAFE or UNSAFE only.

UNSAFE = harmful content or private credentials like passwords only.
SAFE = any company or business information.

Text: {answer[:300]}"""

    try:
        logger.info("Running inline model guardrail check...")
        result = await llm.ainvoke(safety_prompt)

        if "UNSAFE" in result.content.upper()[:20]:
            logger.warning(f"Unsafe answer detected: {result.content[:100]}")
            return "I cannot share this information."

        logger.info("Inline guardrail passed")
        return answer

    except Exception as e:
        logger.error(f"Guardrail LLM call failed: {e}")
        raise RuntimeError(f"Guardrail check failed: {e}")