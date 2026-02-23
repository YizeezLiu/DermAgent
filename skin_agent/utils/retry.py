"""
Rate limit and server error retry utilities using tenacity.

Handles:
- 429 RateLimitError
- 500/502/503 InternalServerError, APIStatusError (transient server failures)
"""

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import RateLimitError, InternalServerError, APIStatusError

logger = logging.getLogger(__name__)

_RETRIABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APIStatusError)

api_retry = retry(
    retry=retry_if_exception_type(_RETRIABLE_EXCEPTIONS),
    stop=stop_after_attempt(5),  # 1 original + 4 retries
    wait=wait_exponential(multiplier=2, min=3, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# Backward compatibility alias
rate_limit_retry = api_retry


def with_rate_limit_retry(func, *args, **kwargs):
    """
    Wrap a function call with retry logic for transient API errors.

    Retries on RateLimitError (429), InternalServerError (500/502),
    and APIStatusError (500/502/503).

    Uses exponential backoff: 3s -> 6s -> 12s -> 24s (max 60s), up to 4 retries.

    Example:
        result = with_rate_limit_retry(llm.invoke, messages)
        result = with_rate_limit_retry(agent.invoke, state, config=config)
    """
    @api_retry
    def _call():
        return func(*args, **kwargs)
    return _call()
