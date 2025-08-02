"""
Rate limiting utilities for API calls
"""

import asyncio
import time
from collections import deque
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter that enforces both requests-per-minute and concurrent request limits
    """
    
    def __init__(self, max_requests_per_minute: int = 60, max_concurrent_requests: int = 10):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent_requests = max_concurrent_requests
        
        # Track request timestamps for rate limiting
        self.request_timestamps = deque()
        
        # Semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        logger.info(f"🚦 Rate limiter initialized: {max_requests_per_minute} req/min, {max_concurrent_requests} concurrent")
    
    async def acquire(self):
        """
        Acquire permission to make a request, enforcing both rate and concurrency limits
        """
        # First, acquire semaphore for concurrent limiting
        await self.semaphore.acquire()
        
        try:
            # Then check rate limiting
            await self._enforce_rate_limit()
            return RateLimitContext(self)
        except Exception:
            # If rate limiting fails, release semaphore
            self.semaphore.release()
            raise
    
    async def _enforce_rate_limit(self):
        """
        Enforce requests-per-minute rate limiting
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # Calculate time to wait until oldest request expires
            oldest_request = self.request_timestamps[0]
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                logger.warning(f"🚦 Rate limit reached ({len(self.request_timestamps)}/{self.max_requests_per_minute}). Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                # Recursively try again after waiting
                await self._enforce_rate_limit()
                return
        
        # Record this request timestamp
        self.request_timestamps.append(current_time)
        logger.debug(f"🚦 Rate limit check passed ({len(self.request_timestamps)}/{self.max_requests_per_minute} in last minute)")


class RateLimitContext:
    """
    Context manager for rate-limited operations
    """
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Release the semaphore when done
        self.rate_limiter.semaphore.release()


class APIRateLimitManager:
    """
    Manages rate limiters for different API providers
    """
    
    def __init__(self, config):
        self.config = config
        
        # Create separate rate limiters for each provider to avoid interference
        base_rate = config.api.max_requests_per_minute
        base_concurrent = config.api.max_concurrent_requests
        
        # Distribute limits across providers (assuming roughly equal usage)
        provider_rate = max(1, base_rate // 3)  # Divide among 3 providers
        provider_concurrent = max(1, base_concurrent // 3)
        
        self.limiters = {
            "openai": RateLimiter(provider_rate, provider_concurrent),
            "anthropic": RateLimiter(provider_rate, provider_concurrent), 
            "google": RateLimiter(provider_rate, provider_concurrent)
        }
        
        logger.info(f"📊 API rate limiters created with {provider_rate} req/min, {provider_concurrent} concurrent per provider")
    
    async def acquire(self, provider: str):
        """
        Acquire rate limit permission for a specific provider
        """
        if provider not in self.limiters:
            # Default limiter for unknown providers
            provider = "openai"
        
        return await self.limiters[provider].acquire() 