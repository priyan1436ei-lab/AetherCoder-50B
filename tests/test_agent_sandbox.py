"""Security and sandbox unit tests for PermissionGuard, Patcher, and FileOps."""

import os
import pytest
import tempfile
from coding_agent.sandbox.permission_guard import PermissionGuard
from coding_agent.tools.file_ops import SandboxedFileOps
from coding_agent.tools.patcher import CodePatcher


def test_permission_guard_blocks_path_traversal():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        guard = PermissionGuard(tmp_workspace)

        # Valid relative file
        valid_path = guard.validate_path("src/main.py")
        assert valid_path.startswith(tmp_workspace)

        # Attempt path traversal escape (../)
        with pytest.raises(PermissionError):
            guard.validate_path("../../windows/system32/cmd.exe")

        with pytest.raises(PermissionError):
            guard.validate_path("../outside.txt")


def test_permission_guard_blocks_sensitive_patterns():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        guard = PermissionGuard(tmp_workspace)

        with pytest.raises(PermissionError):
            guard.validate_path(".ssh/id_rsa")

        with pytest.raises(PermissionError):
            guard.validate_path(".env")


def test_permission_guard_command_allowlist():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        guard = PermissionGuard(tmp_workspace)

        # Allowed command
        guard.validate_command(["pytest", "tests/"])
        guard.validate_command(["python", "script.py"])

        # Disallowed binary
        with pytest.raises(PermissionError):
            guard.validate_command(["curl", "http://malicious.site"])

        # Destructive command
        with pytest.raises(PermissionError):
            guard.validate_command(["python", "-c", "rm -rf /"])


def test_file_ops_and_code_patcher():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        file_ops = SandboxedFileOps(tmp_workspace)
        patcher = CodePatcher(file_ops)

        # Write initial file
        initial_code = "def hello():\n    return 'world'\n"
        file_ops.write_file("hello.py", initial_code)
        assert file_ops.read_file("hello.py") == initial_code

        # Apply surgical replacement
        success, diff = patcher.apply_replacement("hello.py", "'world'", "'antigravity'")
        assert success is True
        assert "-    return 'world'" in diff
        assert "+    return 'antigravity'" in diff

        updated = file_ops.read_file("hello.py")
        assert "antigravity" in updated
