"""Security boundary and permission firewall for sandboxed coding agent execution."""

import os
import re
from typing import List, Optional, Set

# Allowed command binaries for sandboxed execution
COMMAND_ALLOWLIST: Set[str] = {
    "python",
    "python3",
    "pytest",
    "ruff",
    "flake8",
    "mypy",
    "npm",
    "node",
    "cargo",
    "git",
}

# Forbidden command fragments that could compromise host integrity
FORBIDDEN_COMMAND_PATTERNS: List[re.Pattern] = [
    re.compile(r"\brm\s+-[rf]{1,2}\s+\/", re.IGNORECASE),
    re.compile(r"\bdel\s+\/[sfq]", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"curl\s+.*\|\s*(?:bash|sh|powershell|cmd)", re.IGNORECASE),
    re.compile(r"wget\s+.*\|\s*(?:bash|sh|powershell|cmd)", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
]

FORBIDDEN_FILE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\.ssh", re.IGNORECASE),
    re.compile(r"id_rsa", re.IGNORECASE),
    re.compile(r"\.env(?:\.local|\.production)?$", re.IGNORECASE),
    re.compile(r"windows\\system32", re.IGNORECASE),
    re.compile(r"\/etc\/(?:passwd|shadow)", re.IGNORECASE),
]


class PermissionGuard:
    """Enforces strict filesystem confinement and command sandboxing."""

    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)

    def validate_path(self, target_path: str) -> str:
        """Verifies that the target path resolves strictly inside workspace_root."""
        abs_path = os.path.abspath(os.path.join(self.workspace_root, target_path))

        # Check path traversal
        common_prefix = os.path.commonpath([self.workspace_root, abs_path])
        if common_prefix != self.workspace_root:
            raise PermissionError(
                f"Security Violation: Target path '{target_path}' is outside the authorized workspace '{self.workspace_root}'."
            )

        # Check sensitive file patterns
        for pattern in FORBIDDEN_FILE_PATTERNS:
            if pattern.search(abs_path):
                raise PermissionError(
                    f"Security Violation: Access to sensitive file '{os.path.basename(abs_path)}' is blocked by policy."
                )

        return abs_path

    def validate_command(self, cmd_args: List[str]) -> None:
        """Validates that a shell command is within the approved allowlist and safe to execute."""
        if not cmd_args:
            raise ValueError("Empty command cannot be executed.")

        binary_name = os.path.basename(cmd_args[0]).lower().replace(".exe", "")
        if binary_name not in COMMAND_ALLOWLIST:
            raise PermissionError(
                f"Security Violation: Binary '{binary_name}' is not in the approved command allowlist: {sorted(list(COMMAND_ALLOWLIST))}"
            )

        full_cmd_str = " ".join(cmd_args)
        for pattern in FORBIDDEN_COMMAND_PATTERNS:
            if pattern.search(full_cmd_str):
                raise PermissionError(
                    f"Security Violation: Command contains dangerous or destructive instructions matching pattern."
                )
