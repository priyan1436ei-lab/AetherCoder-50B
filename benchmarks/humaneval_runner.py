"""Execution-based HumanEval benchmark harness with sandboxed execution."""

from dataclasses import dataclass
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional


@dataclass
class EvalProblem:
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    error_message: Optional[str] = None


class HumanEvalRunner:
    """Evaluates Python code solutions safely via execution testing."""

    def __init__(self, timeout_seconds: int = 5):
        self.timeout = timeout_seconds

    def check_solution(self, problem: EvalProblem, completion: str) -> EvalResult:
        """Executes the completion concatenated with test assertions in an isolated process."""
        full_code = f"{problem.prompt}\n{completion}\n\n{problem.test}\n\ncheck({problem.entry_point})\n"

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(full_code)
            temp_path = f.name

        try:
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            passed = (res.returncode == 0)
            error = None if passed else (res.stderr or res.stdout)
            return EvalResult(task_id=problem.task_id, passed=passed, error_message=error)
        except subprocess.TimeoutExpired:
            return EvalResult(task_id=problem.task_id, passed=False, error_message=f"Timeout after {self.timeout}s")
        except Exception as exc:
            return EvalResult(task_id=problem.task_id, passed=False, error_message=str(exc))

    def evaluate_suite(self, problems: List[EvalProblem], completions: Dict[str, str]) -> Dict[str, float]:
        """Runs evaluation over a collection of problems and computes pass@1 score."""
        total = len(problems)
        passed_count = 0

        for prob in problems:
            completion = completions.get(prob.task_id, "")
            res = self.check_solution(prob, completion)
            if res.passed:
                passed_count += 1

        pass_at_1 = (passed_count / total) if total > 0 else 0.0
        return {
            "total_problems": total,
            "passed": passed_count,
            "pass_at_1": round(pass_at_1, 4),
        }
