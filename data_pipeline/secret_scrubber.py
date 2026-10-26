"""Secret detection and credential scrubber to prevent sensitive information leakage into the model."""

import math
import re
from typing import List, Tuple

REDACTION_TOKEN = "<SECRET_REDACTED>"

# High confidence regex signatures for cloud, database, and auth tokens
SECRET_PATTERNS = [
    # AWS Access Key ID
    (re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"), "aws_access_key"),
    # GitHub Personal Access Token (classic & fine-grained)
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}\b"), "github_pat"),
    # Generic Private Keys (RSA, OpenSSH, PGP, EC)
    (re.compile(r"-----BEGIN\s+(?:RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE KEY[\s\S]+?-----END\s+(?:RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE KEY-----"), "private_key"),
    # Slack API Tokens
    (re.compile(r"\bxox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*\b"), "slack_token"),
    # Google Cloud API Key
    (re.compile(r"\bAIza[0-9A-Za-z\\-_]{35}\b"), "google_api_key"),
    # OpenAI API Key
    (re.compile(r"\bsk-[a-zA-Z0-9]{20,48}\b"), "openai_api_key"),
    # Hardcoded passwords in connection strings (e.g., postgresql://user:pass@host:5432/db)
    (re.compile(r"(?i)(?:postgres|mysql|mongodb|redis|amqp):\/\/[^:\s]+:([^@\s]+)@"), "db_password"),
]


class SecretScrubber:
    """Detects and redacts credentials, private keys, and high-entropy sensitive tokens."""

    def __init__(self, entropy_threshold: float = 4.5):
        self.entropy_threshold = entropy_threshold

    @staticmethod
    def calculate_shannon_entropy(data: str) -> float:
        """Calculates Shannon entropy for string randomness detection."""
        if not data:
            return 0.0
        entropy = 0.0
        length = len(data)
        freq = {}
        for char in data:
            freq[char] = freq.get(char, 0) + 1
        for count in freq.values():
            p_x = count / length
            entropy -= p_x * math.log2(p_x)
        return entropy

    def scrub(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Scrubs text of known secrets and returns the sanitized text plus detected secret types."""
        findings: List[Tuple[str, str]] = []
        scrubbed = text

        for pattern, secret_type in SECRET_PATTERNS:
            def replace_match(match):
                matched_val = match.group(0)
                findings.append((secret_type, matched_val[:4] + "***"))
                return REDACTION_TOKEN

            scrubbed = pattern.sub(replace_match, scrubbed)

        return scrubbed, findings

    def contains_unscrubbed_secret(self, text: str) -> bool:
        """Quick Boolean check for secret contamination."""
        for pattern, _ in SECRET_PATTERNS:
            if pattern.search(text):
                return True
        return False
