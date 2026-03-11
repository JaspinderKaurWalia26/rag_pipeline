import json
import time
from hashlib import sha256
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from redis.asyncio import Redis

from src.api.schemas import Query
from src.utils.guardrails import validate_input, inline_model_guardrail
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Cache TTL in seconds — cached answers expire after 5 minutes
CACHE_TTL = 300

# Rate limiter — key by client IP address
limiter = Limiter(key_func=get_remote_address)


def build_cache_key(question: str) -> str:
    """
    Build a deterministic SHA-256 Redis key from the question.
    Normalized to lowercase and stripped so 'Hello?' and 'hello?'
    resolve to the same cache entry.
    """
    normalized = question.strip().lower()
    return f"rag-cache:{sha256(normalized.encode()).hexdigest()}"


@router.get("/health")
def health_check():
    """Lightweight health check for load balancer and uptime monitoring."""
    return {"status": "ok"}


@router.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, query: Query):
    """
    Main endpoint for question answering via RAG pipeline.

    Request flow:
    1. Validate input          — reject XSS, HTML injection, prompt injection
    2. Redis cache check       — return immediately on hit, skip LLM entirely
    3. RAG pipeline            — embed → retrieve → async LLM call
    4. Inline guardrail        — async safety check on answer
    5. Cache safe answer       — store guardrail-verified answer in Redis
    6. Return response         — includes per-stage timing breakdown
    """
    try:
        total_start = time.perf_counter()
        logger.info(f"Received question: {query.question[:50]}")

        # Step 1: Validate input — rejects XSS, HTML tags, prompt injection, SQL injection
        # Raises ValueError with user-friendly message if dangerous pattern detected
        clean_question = validate_input(query.question)

        # Step 2: Check Redis cache — skip LLM and guardrail entirely on hit
        cache_key = build_cache_key(clean_question)
        redis: Redis = request.app.state.redis

        cache_start = time.perf_counter()
        cached = await redis.get(cache_key)
        cache_elapsed = round(time.perf_counter() - cache_start, 3)

        if cached:
            logger.info(f"Cache HIT — returned in {cache_elapsed}s")
            return {
                "answer": json.loads(cached),
                "cached": True,
                "timing_seconds": {
                    "cache_lookup": cache_elapsed
                }
            }

        # Step 3: Cache miss — run full async RAG pipeline
        # embed query → retrieve top-2 chunks → call Ollama via ainvoke
        logger.info("Cache MISS — running RAG pipeline")
        rag_chain = request.app.state.rag_chain
        rag_start = time.perf_counter()
        answer = await rag_chain(clean_question)
        rag_elapsed = round(time.perf_counter() - rag_start, 2)
        logger.info(f"RAG pipeline completed in {rag_elapsed}s")

        # Step 4: Async inline guardrail — checks answer for PII, sensitive
        # data, harmful or misleading content before returning to user
        logger.info("Running inline model guardrail...")
        llm = request.app.state.llm
        guardrail_start = time.perf_counter()
        safe_answer = await inline_model_guardrail(answer, llm)
        guardrail_elapsed = round(time.perf_counter() - guardrail_start, 2)
        logger.info(f"Guardrail completed in {guardrail_elapsed}s")

        # Step 5: Cache the safe answer — not the raw answer
        # Ensures cache hits always return guardrail-verified responses
        await redis.set(cache_key, json.dumps(safe_answer), ex=CACHE_TTL)

        total_elapsed = round(time.perf_counter() - total_start, 2)
        logger.info(
            f"Request completed in {total_elapsed}s "
            f"(rag={rag_elapsed}s, guardrail={guardrail_elapsed}s)"
        )

        # Step 6: Return answer with per-stage timing breakdown
        return {
            "answer": safe_answer,
            "cached": False,
            "timing_seconds": {
                "rag_pipeline": rag_elapsed,        # embed + retrieve + LLM
                "guardrail":    guardrail_elapsed,   # safety check LLM call
                "total":        total_elapsed        # full end-to-end time
            }
        }

    except ValueError as e:
        # Raised by validate_input (XSS/injection detected) or empty query
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        # Raised by LLM failure, ChromaDB error, or guardrail failure
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
