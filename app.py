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
    - Stores Redis client, LLM and RAG chain on app.state for endpoint access
    - Initializes FastAPICache with Redis backend
    - Warms up Ollama to absorb cold-start model loading before first user request
    - Closes Redis connection on shutdown
    """
    # Redis — connect and verify
    redis = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True
    )
    pong = await redis.ping()
    logger.info(f"Redis connection verified. PING: {pong}")

    # Store on app.state — accessible in routes via request.app.state
    app.state.redis = redis
    FastAPICache.init(RedisBackend(redis), prefix="rag-cache")

    # LLM initialized
    llm = ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        num_predict=120,
        num_ctx=1024,
        num_thread=4,
    )
    app.state.llm = llm

    # Retriever — connects to ChromaDB persistent vector store
    retriever = ChromaRetriever(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path
    )

    # RAG chain 
    app.state.rag_chain = create_rag_chain(retriever, llm, top_k=2)

    # Warmup — Ollama loads model into RAM on first call 
    # Dummy call at startup ensures users never hit this penalty
    logger.info("Warming up LLM")
    try:
        await llm.ainvoke("hi")
        logger.info("LLM warmup complete")
    except Exception as e:
        logger.warning(f"LLM warmup failed (non-critical): {e}")

    yield

    # Shutdown — close Redis connection cleanly
    await redis.close()
    logger.info("Redis connection closed.")


# FastAPI app 
app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="RAG based question answering API",
    lifespan=lifespan
)

# Response time middleware — adds X-Response-Time header to every response
app.middleware("http")(add_response_time)

# Rate limiter — single instance registered on app.state
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register all routes from src/api/routes/ask.py
app.include_router(router)



@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Returns clean 429 JSON response when rate limit is exceeded."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )
