import json
import time
from hashlib import sha256
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from redis.asyncio import Redis
from config import settings

from src.api.schemas import Query
from src.utils.guardrails import validate_input, inline_model_guardrail, rewrite_query
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

limiter = Limiter(key_func=get_remote_address)


def build_cache_key(question: str) -> str:
    normalized = question.strip().lower()
    return f"rag-cache:{sha256(normalized.encode()).hexdigest()}"


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/ask")
@limiter.limit(settings.rate_limit)
async def ask(request: Request, query: Query):
    try:
        total_start = time.perf_counter()
        logger.info(f"Received question: {query.question[:50]}")

        # Step 1: Validate input
        validated = validate_input(query.question)
        original_question = validated["query"]

        # Step 2: LLM query rewrite 
        rewrite_llm = request.app.state.rewrite_llm
        rewrite_start = time.perf_counter()
        clean_question, was_corrected = await rewrite_query(original_question, rewrite_llm)
        rewrite_elapsed = round(time.perf_counter() - rewrite_start, 2)

        correction_note = (
            f"Did you mean: \"{clean_question}\"? Answering based on corrected query."
            if was_corrected else None
        )

        if was_corrected:
            logger.info(
                f"Query corrected: '{original_question}' to '{clean_question}' "
                f"in {rewrite_elapsed}s"
            )

        # Step 3: Check Redis cache
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
                "correction_note": correction_note,
                "timing_seconds": {
                    "rewrite":      rewrite_elapsed,
                    "cache_lookup": cache_elapsed
                }
            }

        # Step 4: RAG pipeline 
        logger.info("Cache MISS — running RAG pipeline")
        rag_chain = request.app.state.rag_chain
        rag_start = time.perf_counter()
        answer = await rag_chain(clean_question)
        rag_elapsed = round(time.perf_counter() - rag_start, 2)
        logger.info(f"RAG pipeline completed in {rag_elapsed}s")

        # Step 5: Guardrail 
        logger.info("Running inline model guardrail...")
        guardrail_llm = request.app.state.guardrail_llm
        guardrail_start = time.perf_counter()
        safe_answer = await inline_model_guardrail(answer, guardrail_llm)
        guardrail_elapsed = round(time.perf_counter() - guardrail_start, 2)
        logger.info(f"Guardrail completed in {guardrail_elapsed}s")

        # Step 6: Cache the safe answer
        await redis.set(cache_key, json.dumps(safe_answer), ex=settings.cache_ttl)

        total_elapsed = round(time.perf_counter() - total_start, 2)
        logger.info(
            f"Request completed in {total_elapsed}s "
            f"(rewrite={rewrite_elapsed}s, rag={rag_elapsed}s, guardrail={guardrail_elapsed}s)"
        )

        # Step 7: Return
        return {
            "answer": safe_answer,
            "cached": False,
            "correction_note": correction_note,
            "timing_seconds": {
                "rewrite":      rewrite_elapsed,
                "rag_pipeline": rag_elapsed,
                "guardrail":    guardrail_elapsed,
                "total":        total_elapsed
            }
        }

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")