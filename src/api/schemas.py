from pydantic import BaseModel, Field


class Query(BaseModel):
    """Request schema for the /ask endpoint."""
    question: str = Field(
        min_length=3,
        max_length=500,
        description="User question"
    )


class TimingResponse(BaseModel):
    """Timing breakdown returned in every response."""
    rag_pipeline: float | None = None    # embed + retrieve + LLM
    guardrail: float | None = None       # safety check LLM call
    total: float | None = None           # full end-to-end time
    cache_lookup: float | None = None    # only on cache hit


class AnswerResponse(BaseModel):
    """Response schema for the /ask endpoint."""
    answer: str
    cached: bool
    timing_seconds: TimingResponse
