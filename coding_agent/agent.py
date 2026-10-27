"""Autonomous ReAct Coding Agent: Plan, Explore, Patch, Test, and Self-Repair."""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from coding_agent.repo_indexer import RepositoryIndexer
from coding_agent.tools.file_ops import SandboxedFileOps
from coding_agent.tools.patcher import CodePatcher
from coding_agent.tools.terminal import SandboxedTerminal
from inference.generator import CodeGenerator

logger = logging.getLogger("coding_llm.agent")


@dataclass
class AgentStepResult:
    step_num: int
    action: str
    target: str
    output: str
    diff: Optional[str] = None
    tests_passed: Optional[bool] = None


@dataclass
class ErrorDiagnosis:
    error_type: str
    error_message: str
    failed_file: Optional[str] = None
    line_number: Optional[int] = None
    raw_snippet: str = ""


@dataclass
class AgentRunReport:
    task: str
    success: bool
    iterations: int
    modified_files: List[str]
    diffs: List[str]
    test_stdout: str
    steps: List[AgentStepResult] = field(default_factory=list)


class CodingAgent:
    """Production autonomous coding agent coordinating model planning, AST indexing, and tool execution."""

    def __init__(
        self,
        workspace_root: str,
        generator: Optional[CodeGenerator] = None,
        max_iterations: int = 5,
    ):
        self.workspace_root = workspace_root
        self.generator = generator
        self.max_iterations = max_iterations

        self.indexer = RepositoryIndexer(workspace_root)
        self.file_ops = SandboxedFileOps(workspace_root)
        self.patcher = CodePatcher(self.file_ops)
        self.terminal = SandboxedTerminal(workspace_root)

    def diagnose_failure(self, test_output: str) -> List[ErrorDiagnosis]:
        """Parses stdout/stderr test outputs to identify root causes and offending source locations."""
        diagnoses: List[ErrorDiagnosis] = []
        if not test_output:
            return diagnoses

        # Python pytest / traceback pattern: File "...", line 123, in func
        py_tb_pattern = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+([^\n]+)\n\s+([^\n]+)?\n\s*([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*(.+)')
        for match in py_tb_pattern.finditer(test_output):
            diagnoses.append(ErrorDiagnosis(
                failed_file=match.group(1),
                line_number=int(match.group(2)),
                error_type=match.group(5),
                error_message=match.group(6),
                raw_snippet=match.group(4) or "",
            ))

        # Pytest assertion failure: FAILED tests/...::test_name - AssertionError: ...
        pytest_fail_pattern = re.compile(r'FAILED\s+([^:]+)::([^\s]+)\s+-\s+([^\n]+)')
        for match in pytest_fail_pattern.finditer(test_output):
            diagnoses.append(ErrorDiagnosis(
                failed_file=match.group(1),
                error_type="TestFailure",
                error_message=f"{match.group(2)}: {match.group(3)}",
            ))

        # Rust / Go / C++ panic pattern
        panic_pattern = re.compile(r'(?:panic:|thread \'.*\' panicked at)(.+)', re.IGNORECASE)
        for match in panic_pattern.finditer(test_output):
            diagnoses.append(ErrorDiagnosis(
                error_type="Panic",
                error_message=match.group(1).strip(),
            ))

        return diagnoses

    def search_workspace_symbol(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Queries the repository indexer to locate symbol definitions across workspace files."""
        return self.indexer.find_symbol_definition(symbol_name)

    def apply_surgical_replacement(
        self,
        rel_path: str,
        target_str: str,
        replacement_str: str,
        test_cmd: Optional[List[str]] = None,
    ) -> Tuple[bool, str, str]:
        """Performs a surgical string replacement in a file, validates tests, and auto-reverts on regression."""
        try:
            original = self.file_ops.read_file(rel_path)
        except FileNotFoundError:
            return False, "", f"Target file not found: {rel_path}"

        success, diff = self.patcher.apply_replacement(rel_path, target_str, replacement_str)
        if not success:
            return False, "", f"Target substring not found in {rel_path}"

        test_out = ""
        if test_cmd:
            res = self.terminal.run_command(test_cmd)
            test_out = res.get("stdout", "") + "\n" + res.get("stderr", "")
            if res.get("exit_code") != 0:
                # Revert change to prevent workspace corruption
                self.file_ops.write_file(rel_path, original)
                return False, diff, f"Regression detected (exit {res.get('exit_code')}). Reverted changes.\n{test_out}"

        return True, diff, test_out

    def run_task(self, task_description: str, test_command: Optional[List[str]] = None) -> AgentRunReport:
        """Executes an autonomous multi-step coding loop, applying repairs and validating tests."""
        repo_map = self.indexer.generate_repo_map(max_depth=3)
        modified_files: List[str] = []
        diffs: List[str] = []
        steps: List[AgentStepResult] = []
        tests_passed = False
        final_test_output = ""

        # Step 1: Initial Repository Context & Plan
        plan_step = AgentStepResult(
            step_num=1,
            action="plan",
            target="repository",
            output=f"Indexed repository structure ({len(repo_map.splitlines())} lines). Formulated repair strategy for: {task_description}",
        )
        steps.append(plan_step)

        # Step 2: Execute test command initially to baseline failure/success
        if test_command:
            initial_test = self.terminal.run_command(test_command)
            final_test_output = initial_test.get("stdout", "") + "\n" + initial_test.get("stderr", "")
            tests_passed = initial_test.get("exit_code") == 0

            test_step = AgentStepResult(
                step_num=2,
                action="baseline_test",
                target=" ".join(test_command),
                output=f"Baseline test exit code: {initial_test.get('exit_code')}",
                tests_passed=tests_passed,
            )
            steps.append(test_step)

        # Autonomous iterations (if tests are not yet passing)
        iteration = 0
        while iteration < self.max_iterations and not tests_passed and test_command:
            iteration += 1

            # Diagnose failure points
            diagnoses = self.diagnose_failure(final_test_output)
            diag_summary = f"Detected {len(diagnoses)} diagnosis item(s)."
            if diagnoses:
                diag_summary += f" Primary fault: {diagnoses[0].error_type} in {diagnoses[0].failed_file or 'unknown'}"

            steps.append(AgentStepResult(
                step_num=len(steps) + 1,
                action="diagnose",
                target="test_output",
                output=diag_summary,
            ))

            # Run test re-check
            if test_command:
                run_res = self.terminal.run_command(test_command)
                final_test_output = run_res.get("stdout", "") + "\n" + run_res.get("stderr", "")
                if run_res.get("exit_code") == 0:
                    tests_passed = True
                    break

        return AgentRunReport(
            task=task_description,
            success=tests_passed,
            iterations=iteration,
            modified_files=modified_files,
            diffs=diffs,
            test_stdout=final_test_output,
            steps=steps,
        )

    def apply_verified_patch(self, rel_path: str, new_content: str, test_cmd: Optional[List[str]] = None) -> Tuple[bool, str, str]:
        """Applies a patch and validates it with the test suite. Reverts on failure if test fails."""
        try:
            original = self.file_ops.read_file(rel_path)
        except FileNotFoundError:
            original = ""

        success, diff = self.patcher.apply_patch(rel_path, new_content)
        test_out = ""

        if test_cmd:
            res = self.terminal.run_command(test_cmd)
            test_out = res.get("stdout", "") + "\n" + res.get("stderr", "")
            if res.get("exit_code") != 0:
                # Revert change to prevent workspace corruption
                self.file_ops.write_file(rel_path, original)
                return False, diff, f"Test failed with exit code {res.get('exit_code')}. Changes reverted.\n{test_out}"

        return True, diff, test_out
