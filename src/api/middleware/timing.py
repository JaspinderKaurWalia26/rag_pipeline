import time
from fastapi import Request
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def add_response_time(request: Request, call_next):
    """
    Middleware to track end-to-end response time for every request.
    - Uses time.perf_counter() for high-resolution timing
    - Adds X-Response-Time header to response (visible in Postman / curl)
    - Logs the duration alongside other request logs
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round(time.perf_counter() - start, 2)

    # Attach timing as response header — visible in Postman under Headers tab
    response.headers["X-Response-Time"] = f"{elapsed}s"
    logger.info(f"{request.method} {request.url.path} completed in {elapsed}s")

    return response
