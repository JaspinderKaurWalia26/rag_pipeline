from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from langchain_ollama import ChatOllama
from redis.asyncio import Redis
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.api.routes.ask import router
from src.api.middleware.timing import add_response_time
from src.core.retriever import ChromaRetriever
from src.core.rag_pipeline import create_rag_chain
from src.utils.logger import setup_logger
from config import settings

logger = setup_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown events:
    - Connects to Redis and verifies connection
    - Initializes three separate LLM instances for RAG, guardrail, and rewrite
    - Stores all instances on app.state for endpoint access
    - Initializes FastAPICache with Redis backend
    - Warms up all three models to absorb cold-start penalty before first request
    - Closes Redis connection on shutdown
    """

    # Connect to Redis and verify the connection is alive
    redis = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True
    )
    pong = await redis.ping()
    logger.info(f"Redis connection verified. PING: {pong}")

    # Store Redis client on app.state — accessible in all routes
    app.state.redis = redis

    # Initialize FastAPICache with Redis as the backend
    # All cached responses will be stored under the "rag-cache" prefix
    FastAPICache.init(RedisBackend(redis), prefix="rag-cache")

    # Main LLM — used only for RAG answer generation
    llm = ChatOllama(
         model="qwen2.5:1.5b",
        temperature=settings.llm_temperature,
        num_predict=settings.llm_num_predict,
        num_ctx=settings.llm_num_ctx,
        num_thread=settings.llm_num_thread,
    )
    app.state.llm = llm

    # Guardrail LLM — used only for safety checks on generated answers
    guardrail_llm = ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0,
        num_predict=10,
    )
    app.state.guardrail_llm = guardrail_llm

    # Rewrite LLM 
    rewrite_llm = ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0,
        num_predict=50,
    )
    app.state.rewrite_llm = rewrite_llm

    # Initialize ChromaDB retriever — connects to persistent vector store on disk
    retriever = ChromaRetriever(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path
    )

    # Build the RAG chain — combines retriever and main LLM into a reusable pipeline
    app.state.rag_chain = create_rag_chain(retriever, llm, top_k=settings.top_k_results)

    # Warmup main LLM — Ollama loads the model into RAM on the first call
    logger.info("Warming up main LLM")
    try:
        await llm.ainvoke("hi")
        logger.info("Main LLM warmup complete")
    except Exception as e:
        logger.warning(f"Main LLM warmup failed (non-critical): {e}")

    # Warmup guardrail LLM 
    logger.info("Warming up guardrail LLM")
    try:
        await guardrail_llm.ainvoke("hi")
        logger.info("Guardrail LLM warmup complete")
    except Exception as e:
        logger.warning(f"Guardrail LLM warmup failed (non-critical): {e}")

    # Warmup rewrite LLM
    logger.info("Warming up rewrite LLM")
    try:
        await rewrite_llm.ainvoke("hi")
        logger.info("Rewrite LLM warmup complete")
    except Exception as e:
        logger.warning(f"Rewrite LLM warmup failed (non-critical): {e}")

    yield

    # Shutdown — close Redis connection cleanly
    await redis.close()
    logger.info("Redis connection closed.")


# Initialize FastAPI application
app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="RAG based question answering API",
    lifespan=lifespan
)

# Response time middleware — adds X-Response-Time header to every response
app.middleware("http")(add_response_time)

# Rate limiter — limits requests per IP address to prevent abuse
# Single instance registered on app.state and used across all routes
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# CORS middleware — controls which origins are allowed to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register all routes defined in src/api/routes/ask.py
app.include_router(router)


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Returns a clean 429 JSON response when the rate limit is exceeded."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )