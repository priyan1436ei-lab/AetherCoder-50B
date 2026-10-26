"""License detection, provenance tracking, and permissive usage filter for code pretraining."""

import re
from typing import Dict, Optional, Set, Tuple

PERMISSIVE_LICENSES: Set[str] = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
    "0bsd",
}

RESTRICTED_OR_COPYLEFT: Set[str] = {
    "gpl-2.0",
    "gpl-3.0",
    "agpl-3.0",
    "lgpl-2.1",
    "lgpl-3.0",
    "sspl",
    "proprietary",
}

# Fast keyword fingerprints for permissive licenses
LICENSE_PATTERNS: Dict[str, re.Pattern] = {
    "mit": re.compile(r"mit\s+license|permission is hereby granted,\s+free of charge", re.IGNORECASE),
    "apache-2.0": re.compile(r"apache\s+license[,\s]+version\s+2\.0", re.IGNORECASE),
    "bsd-3-clause": re.compile(r"redistribution and use in source and binary forms|neither the name of .* nor the names of its contributors", re.IGNORECASE),
    "bsd-2-clause": re.compile(r"redistributions of source code must retain the above copyright notice", re.IGNORECASE),
    "gpl-3.0": re.compile(r"gnu general public license\s+version\s+3|gplv3", re.IGNORECASE),
    "agpl-3.0": re.compile(r"gnu affero general public license|agplv3", re.IGNORECASE),
}


class LicenseFilter:
    """Enforces intellectual property boundaries and permissibility filtering on training data."""

    def __init__(self, allowed_licenses: Optional[Set[str]] = None):
        self.allowed_licenses = allowed_licenses or PERMISSIVE_LICENSES

    def detect_license_from_text(self, license_text: str) -> Tuple[Optional[str], float]:
        """Detects the license type and confidence score from a license file or header."""
        if not license_text:
            return None, 0.0

        for name, pattern in LICENSE_PATTERNS.items():
            if pattern.search(license_text):
                return name, 0.95

        return "unknown", 0.0

    def is_permissible(self, license_identifier: Optional[str], license_text: Optional[str] = None) -> bool:
        """Determines if the given repository or file is compliant for unrestricted training."""
        if license_identifier:
            norm_id = license_identifier.strip().lower()
            if norm_id in self.allowed_licenses:
                return True
            if norm_id in RESTRICTED_OR_COPYLEFT:
                return False

        if license_text:
            detected, confidence = self.detect_license_from_text(license_text)
            if detected and detected in self.allowed_licenses and confidence >= 0.8:
                return True

        return False
