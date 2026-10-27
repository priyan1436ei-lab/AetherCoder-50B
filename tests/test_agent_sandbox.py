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


def test_multilanguage_indexer_and_symbol_search():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        from coding_agent.repo_indexer import RepositoryIndexer
        file_ops = SandboxedFileOps(tmp_workspace)
        indexer = RepositoryIndexer(tmp_workspace)

        # Write Python, TypeScript, and Go files
        file_ops.write_file("src/math.py", "def compute_factorial(n: int) -> int:\n    return 1\n")
        file_ops.write_file("src/service.ts", "export class AuthService {\n  login() {}\n}\nexport function verifyToken() {}\n")
        file_ops.write_file("src/worker.go", "package main\n\nfunc RunWorker() {}\ntype JobQueue struct {}\n")

        # Index and verify Python
        py_data = indexer.index_file("src/math.py")
        assert any(s["name"] == "compute_factorial" for s in py_data["symbols"])

        # Index and verify TypeScript
        ts_data = indexer.index_file("src/service.ts")
        assert any(s["name"] == "AuthService" for s in ts_data["symbols"])
        assert any(s["name"] == "verifyToken" for s in ts_data["symbols"])

        # Index and verify Go
        go_data = indexer.index_file("src/worker.go")
        assert any(s["name"] == "RunWorker" for s in go_data["symbols"])
        assert any(s["name"] == "JobQueue" for s in go_data["symbols"])

        # Global symbol search
        matches = indexer.find_symbol_definition("AuthService")
        assert len(matches) > 0
        assert matches[0]["file"] == os.path.normpath("src/service.ts")


def test_coding_agent_diagnosis_and_surgical_rollback():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        from coding_agent.agent import CodingAgent
        agent = CodingAgent(tmp_workspace)

        # Test failure diagnosis parser
        mock_output = (
            'File "math_lib.py", line 42, in add\n'
            '    assert result == 10\n'
            'AssertionError: Expected 10 but got 9\n'
        )
        diagnoses = agent.diagnose_failure(mock_output)
        assert len(diagnoses) > 0
        assert diagnoses[0].error_type == "AssertionError"
        assert diagnoses[0].line_number == 42

        # Test patch rollback on test failure
        agent.file_ops.write_file("calc.py", "def add(a, b):\n    return a + b\n")
        # Proposed bad patch that breaks functionality
        bad_code = "def add(a, b):\n    return a - b\n"
        # Test command that fails with exit code 1
        success, diff, out = agent.apply_verified_patch(
            "calc.py",
            bad_code,
            test_cmd=["python", "-c", "import sys; sys.exit(1)"],
        )
        assert success is False
        assert "Changes reverted" in out
        # Verify file was safely restored
        assert agent.file_ops.read_file("calc.py") == "def add(a, b):\n    return a + b\n"


def test_permission_guard_symlink_escape():
    with tempfile.TemporaryDirectory() as outer_dir:
        workspace = os.path.join(outer_dir, "workspace")
        outside = os.path.join(outer_dir, "secret_outside.txt")
        os.makedirs(workspace, exist_ok=True)
        with open(outside, "w") as f:
            f.write("top-secret")

        guard = PermissionGuard(workspace)
        symlink_path = os.path.join(workspace, "escape_link")

        # Create symlink pointing outside workspace (if OS permits)
        try:
            os.symlink(outside, symlink_path)
            with pytest.raises(PermissionError, match="Symlink escape"):
                guard.validate_path("escape_link")
        except (OSError, NotImplementedError):
            # Windows non-admin without Developer Mode might disallow symlink creation
            pass


def test_permission_guard_command_injection_and_unapproved_install():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        guard = PermissionGuard(tmp_workspace)

        # Semicolon injection
        with pytest.raises(PermissionError, match="dangerous shell metacharacter"):
            guard.validate_command(["python", "script.py; rm -rf /"])

        # Pipe injection
        with pytest.raises(PermissionError, match="dangerous shell metacharacter"):
            guard.validate_command(["pytest", "| echo hack"])

        # Unapproved npm install
        with pytest.raises(PermissionError, match="Package installation"):
            guard.validate_command(["npm", "install", "malicious-pkg"])

        # Unapproved pip install
        with pytest.raises(PermissionError, match="pip install requires explicit approval"):
            guard.validate_command(["python", "-m", "pip", "install", "malicious-pkg"])


def test_sandboxed_terminal_env_secrets_stripping():
    with tempfile.TemporaryDirectory() as tmp_workspace:
        from coding_agent.tools.terminal import SandboxedTerminal
        terminal = SandboxedTerminal(tmp_workspace)

        # Set fake secrets in current environment
        os.environ["AETHERCODER_API_KEY"] = "super-secret-key-12345"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "fake-aws-secret"
        os.environ["GITHUB_TOKEN"] = "ghp_faketoken98765"

        try:
            # Query environment from child python process
            res = terminal.run_command([
                "python",
                "-c",
                "import os; print('KEY_IN_ENV=' + str('AETHERCODER_API_KEY' in os.environ)); print('AWS_IN_ENV=' + str('AWS_SECRET_ACCESS_KEY' in os.environ))",
            ])
            assert res["exit_code"] == 0
            assert "KEY_IN_ENV=False" in res["stdout"]
            assert "AWS_IN_ENV=False" in res["stdout"]
        finally:
            del os.environ["AETHERCODER_API_KEY"]
            del os.environ["AWS_SECRET_ACCESS_KEY"]
            del os.environ["GITHUB_TOKEN"]


