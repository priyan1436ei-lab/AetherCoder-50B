"""Authentication, rate limiting, and security auditing for the inference API."""

import logging
import os
import secrets
import time
from typing import Dict, List, Optional
from fastapi import Header, HTTPException, status

logger = logging.getLogger("coding_llm.security")


class SecurityManager:
    """Handles API token verification and per-client token bucket rate limiting."""

    def __init__(
        self,
        master_key: Optional[str] = None,
        rate_limit_rpm: int = 60,
        allow_anonymous: Optional[bool] = None,
    ):
        # Explicit master key takes precedence, fallback to env var
        self.master_key = master_key or os.environ.get("AETHERCODER_API_KEY")
        self.rate_limit_rpm = rate_limit_rpm

        # Anonymous access must be explicitly enabled if master_key is not set
        if allow_anonymous is not None:
            self.allow_anonymous = allow_anonymous
        else:
            self.allow_anonymous = os.environ.get(
                "AETHERCODER_ALLOW_ANONYMOUS", ""
            ).lower() in ("1", "true", "yes")

        self.client_request_timestamps: Dict[str, List[float]] = {}
        self._last_cleanup = time.time()

    def verify_auth_header(self, authorization: Optional[str] = Header(None)) -> str:
        """Validates Bearer token against configured key."""
        if not self.master_key:
            if not self.allow_anonymous:
                logger.error(
                    "API key is not configured and anonymous access is disabled. "
                    "Set AETHERCODER_API_KEY or AETHERCODER_ALLOW_ANONYMOUS=1."
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=(
                        "Authentication required. Server has no API key configured and "
                        "anonymous access is disabled. Set AETHERCODER_API_KEY or "
                        "AETHERCODER_ALLOW_ANONYMOUS=1."
                    ),
                )
            return "anonymous_local"

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization header. Use 'Bearer <API_KEY>'",
            )

        token = authorization.split("Bearer ", 1)[1].strip()
        # Constant-time comparison prevents timing side-channel attacks
        if not secrets.compare_digest(token, self.master_key):
            logger.warning("Failed authentication attempt with invalid token.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key.",
            )

        return token

    def check_rate_limit(self, client_id: str) -> None:
        """Token bucket rate limiter enforcing requests per minute."""
        now = time.time()
        window_start = now - 60.0

        # Periodic cleanup of inactive clients to prevent memory leaks
        if now - self._last_cleanup > 300.0:
            self._cleanup_old_clients(window_start)
            self._last_cleanup = now

        timestamps = self.client_request_timestamps.get(client_id, [])
        timestamps = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.rate_limit_rpm:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({self.rate_limit_rpm} req/min). Please back off.",
            )

        timestamps.append(now)
        self.client_request_timestamps[client_id] = timestamps

    def _cleanup_old_clients(self, window_start: float) -> None:
        to_del = [
            cid
            for cid, times in self.client_request_timestamps.items()
            if not times or times[-1] <= window_start
        ]
        for cid in to_del:
            del self.client_request_timestamps[cid]

