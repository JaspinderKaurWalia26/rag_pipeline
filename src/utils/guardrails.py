import re
from langchain_ollama import ChatOllama
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Dangerous patterns — reject input if any of these are found
# Covers: XSS, HTML injection, prompt injection, SQL injection
REJECT_PATTERNS = re.compile(
    r'<[^>]+'                                        # HTML tags e.g. <script>, <img>
    r'|javascript\s*:'                               # javascript: in URLs
    r'|on\w+\s*='                                    # onclick=, onerror=, onload=
    r'|ignore\s+(previous|all|above)\s+instructions' # prompt injection
    r'|forget\s+(previous|all|above)\s+instructions' # prompt injection variant
    r'|system\s*prompt'                              # trying to access system prompt
    r'|union\s+select|drop\s+table'                  # SQL injection
    r'|;\s*(drop|delete|update|insert)',              # SQL injection variant
    re.IGNORECASE
)


def validate_input(text: str) -> str:
    """
    Validate user input — reject if dangerous pattern found.
    Does not sanitize or clean — rejects completely with 400 error.
    Works for any language — only dangerous patterns are rejected.

    Args:
        text: Raw user input from request
    Returns:
        Stripped input if safe
    Raises:
        ValueError: If input is empty, too long, or contains dangerous content
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
    return text.strip()


async def inline_model_guardrail(answer: str, llm: ChatOllama) -> str:
    """
    Async safety check on the LLM-generated answer.
    Returns original answer if SAFE, fallback message if UNSAFE.
    Raises RuntimeError if the LLM call fails.
    """
    if not answer.strip():
        logger.warning("Empty answer provided to guardrail")
        return "No answer was generated."

    safety_prompt = f"""Evaluate this response for safety. Check for:
1. Sensitive or confidential information leak
2. Personal Identifiable Information (PII)
3. Harmful or inappropriate content
4. Misleading or false information

Reply with only SAFE or UNSAFE followed by a one-line reason.

Response to evaluate:
{answer}"""

    try:
        logger.info("Running inline model guardrail check...")
        result = await llm.ainvoke(safety_prompt)

        if "UNSAFE" in result.content.upper():
            logger.warning(f"Unsafe answer detected: {result.content[:100]}")
            return "I cannot share this information."

        logger.info("Inline guardrail passed")
        return answer

    except Exception as e:
        logger.error(f"Guardrail LLM call failed: {e}")
        raise RuntimeError(f"Guardrail check failed: {e}")