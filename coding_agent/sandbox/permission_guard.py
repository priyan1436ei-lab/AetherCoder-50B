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

# Subcommands that modify global environment or perform unapproved network downloads
PACKAGE_INSTALL_COMMANDS = {
    ("python", "-m", "pip", "install"),
    ("pip", "install"),
    ("npm", "install"),
    ("npm", "i"),
    ("cargo", "install"),
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

# Sensitive environment variables that must NEVER be leaked to subprocesses
SENSITIVE_ENV_PATTERNS = [
    re.compile(r"(?:KEY|SECRET|TOKEN|PASSW|AUTH|CREDENTIAL)", re.IGNORECASE),
    re.compile(r"^(?:AWS_|GITHUB_|OPENAI_|AETHERCODER_)", re.IGNORECASE),
]


class PermissionGuard:
    """Enforces strict filesystem confinement and command sandboxing."""

    def __init__(self, workspace_root: str, allow_network: bool = False, allow_package_install: bool = False):
        self.workspace_root = os.path.abspath(workspace_root)
        self.allow_network = allow_network
        self.allow_package_install = allow_package_install

    def validate_path(self, target_path: str) -> str:
        """Verifies that the target path resolves strictly inside workspace_root, preventing traversal and symlink escapes."""
        abs_path = os.path.abspath(os.path.join(self.workspace_root, target_path))

        # Check path traversal on lexical path
        common_prefix = os.path.commonpath([self.workspace_root, abs_path])
        if common_prefix != self.workspace_root:
            raise PermissionError(
                f"Security Violation: Target path '{target_path}' is outside authorized workspace '{self.workspace_root}'."
            )

        # Check symlink escapes by resolving realpath
        real_root = os.path.realpath(self.workspace_root)
        if os.path.exists(abs_path):
            real_path = os.path.realpath(abs_path)
            if os.path.commonpath([real_root, real_path]) != real_root:
                raise PermissionError(
                    f"Security Violation: Symlink escape detected. '{target_path}' resolves outside workspace root."
                )
        else:
            # For new files, verify existing ancestor resolves inside workspace
            parent = os.path.dirname(abs_path)
            while parent and not os.path.exists(parent):
                parent = os.path.dirname(parent)
            if parent and os.path.exists(parent):
                real_parent = os.path.realpath(parent)
                if os.path.commonpath([real_root, real_parent]) != real_root:
                    raise PermissionError(
                        f"Security Violation: Target directory resolves outside workspace root."
                    )

        # Check sensitive file patterns
        for pattern in FORBIDDEN_FILE_PATTERNS:
            if pattern.search(abs_path):
                raise PermissionError(
                    f"Security Violation: Access to sensitive file '{os.path.basename(abs_path)}' is blocked by policy."
                )

        return abs_path

    def validate_command(self, cmd_args: List[str]) -> None:
        """Validates that a command and its individual arguments are safe to execute."""
        if not cmd_args:
            raise ValueError("Empty command cannot be executed.")

        binary_name = os.path.basename(cmd_args[0]).lower().replace(".exe", "")
        if binary_name not in COMMAND_ALLOWLIST:
            raise PermissionError(
                f"Security Violation: Binary '{binary_name}' is not in the approved command allowlist: {sorted(list(COMMAND_ALLOWLIST))}"
            )

        # Disallow dangerous shell metacharacters in individual arguments
        shell_metachars = ["&&", "||", "|", "`", "$("]
        for i, arg in enumerate(cmd_args):
            for meta in shell_metachars:
                if meta in arg:
                    raise PermissionError(
                        f"Security Violation: Command argument contains dangerous shell metacharacter '{meta}'."
                    )
            if ";" in arg:
                is_code_arg = i > 0 and cmd_args[i - 1] in ("-c", "-e")
                if not is_code_arg:
                    raise PermissionError(
                        "Security Violation: Command argument contains dangerous shell metacharacter ';'."
                    )

            if i > 0 and cmd_args[i - 1] == "-c":
                forbidden_py = ["os.system", "shutil.rmtree", "subprocess", "socket"]
                for bad in forbidden_py:
                    if bad in arg:
                        raise PermissionError(f"Security Violation: Inline script contains forbidden call '{bad}'.")


        # Block unapproved package installation
        if not self.allow_package_install:
            cmd_lower = [a.lower() for a in cmd_args]
            if len(cmd_lower) >= 2 and (
                (cmd_lower[0] == "npm" and cmd_lower[1] in ("install", "i"))
                or (cmd_lower[0] == "cargo" and cmd_lower[1] == "install")
            ):
                raise PermissionError(
                    "Security Violation: Package installation commands require explicit approval."
                )
            if len(cmd_lower) >= 4 and cmd_lower[0] in ("python", "python3") and cmd_lower[1] == "-m" and cmd_lower[2] == "pip" and cmd_lower[3] == "install":
                raise PermissionError(
                    "Security Violation: pip install requires explicit approval."
                )

        # Pattern check across whole command string
        full_cmd_str = " ".join(cmd_args)
        for pattern in FORBIDDEN_COMMAND_PATTERNS:
            if pattern.search(full_cmd_str):
                raise PermissionError(
                    "Security Violation: Command contains dangerous or destructive instructions matching policy pattern."
                )

    def get_sanitized_env(self, custom_env: Optional[dict] = None) -> dict:
        """Generates a clean environment dictionary stripping all sensitive host keys and tokens."""
        safe_keys = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "SYSTEMDRIVE",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "HOME",
            "LANG",
            "LC_ALL",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "COMSPEC",
        }
        clean_env = {}
        for k, v in os.environ.items():
            k_upper = k.upper()
            if k_upper in safe_keys:
                # Still check if the variable itself is a sensitive pattern
                is_sensitive = any(p.search(k) for p in SENSITIVE_ENV_PATTERNS)
                if not is_sensitive:
                    clean_env[k] = v

        if custom_env:
            for k, v in custom_env.items():
                if not any(p.search(k) for p in SENSITIVE_ENV_PATTERNS):
                    clean_env[k] = v

        return clean_env

