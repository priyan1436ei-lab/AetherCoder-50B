"""Sandboxed command runner with timeout guards and permission firewall."""

import subprocess
from typing import Dict, List, Optional
from coding_agent.sandbox.permission_guard import PermissionGuard


class SandboxedTerminal:
    """Executes permitted testing and validation tools inside the project workspace."""

    def __init__(self, workspace_root: str, default_timeout_seconds: int = 60):
        self.guard = PermissionGuard(workspace_root)
        self.workspace_root = self.guard.workspace_root
        self.default_timeout = default_timeout_seconds

    def run_command(
        self,
        cmd_args: List[str],
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        """Runs an authorized command and returns stdout, stderr, and exit code."""
        # Enforce security validation
        self.guard.validate_command(cmd_args)

        timeout_sec = timeout or self.default_timeout

        try:
            proc = subprocess.run(
                cmd_args,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": -1,
                "stdout": exc.stdout or "",
                "stderr": f"Command timed out after {timeout_sec} seconds.",
                "timed_out": True,
            }
        except Exception as exc:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
            }
