"""Authentication, rate limiting, and security auditing for the inference API."""

import logging
import time
from typing import Dict, Optional
from fastapi import Header, HTTPException, status

logger = logging.getLogger("coding_llm.security")


class SecurityManager:
    """Handles API token verification and per-client token bucket rate limiting."""

    def __init__(self, master_key: Optional[str] = None, rate_limit_rpm: int = 60):
        self.master_key = master_key
        self.rate_limit_rpm = rate_limit_rpm
        self.client_request_timestamps: Dict[str, list] = {}

    def verify_auth_header(self, authorization: Optional[str] = Header(None)) -> str:
        """Validates Bearer token against configured key."""
        if not self.master_key:
            return "anonymous_local"

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization header. Use 'Bearer <API_KEY>'",
            )

        token = authorization.split("Bearer ")[1].strip()
        if token != self.master_key:
            logger.warning(f"Failed authentication attempt with token: {token[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key.",
            )

        return token

    def check_rate_limit(self, client_id: str) -> None:
        """Token bucket rate limiter enforcing requests per minute."""
        now = time.time()
        window_start = now - 60.0

        timestamps = self.client_request_timestamps.get(client_id, [])
        # Prune old timestamps
        timestamps = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.rate_limit_rpm:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({self.rate_limit_rpm} req/min). Please back off.",
            )

        timestamps.append(now)
        self.client_request_timestamps[client_id] = timestamps
