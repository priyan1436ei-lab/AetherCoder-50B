"""Execution-based HumanEval benchmark harness with sandboxed execution."""

from dataclasses import asdict, dataclass
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional


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

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(full_code)
                temp_path = f.name

            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            passed = res.returncode == 0
            error = None if passed else (res.stderr or res.stdout)
            return EvalResult(task_id=problem.task_id, passed=passed, error_message=error)
        except subprocess.TimeoutExpired:
            return EvalResult(task_id=problem.task_id, passed=False, error_message=f"Timeout after {self.timeout}s")
        except Exception as exc:
            return EvalResult(task_id=problem.task_id, passed=False, error_message=str(exc))
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def evaluate_suite(self, problems: List[EvalProblem], completions: Dict[str, str]) -> Dict[str, Any]:
        """Runs evaluation over a collection of problems and computes pass@1 score."""
        total = len(problems)
        passed_count = 0
        results: List[EvalResult] = []

        for prob in problems:
            completion = completions.get(prob.task_id, "")
            res = self.check_solution(prob, completion)
            results.append(res)
            if res.passed:
                passed_count += 1

        pass_at_1 = (passed_count / total) if total > 0 else 0.0
        return {
            "total_problems": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_at_1": round(pass_at_1, 4),
            "results": [asdict(r) for r in results],
        }

    def evaluate_and_save_report(
        self,
        problems: List[EvalProblem],
        completions: Dict[str, str],
        output_json_path: str,
        model_name: str = "codex-150m-prototype",
        checkpoint_path: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Evaluates a problem suite and exports an audited, machine-readable JSON report."""
        eval_metrics = self.evaluate_suite(problems, completions)

        report = {
            "benchmark": "HumanEval-Compatible",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "environment": {
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "processor": platform.processor() or "Unknown",
            },
            "summary": {
                "total_problems": eval_metrics["total_problems"],
                "passed": eval_metrics["passed"],
                "failed": eval_metrics["failed"],
                "pass_at_1": eval_metrics["pass_at_1"],
            },
            "notes": notes,
            "details": eval_metrics["results"],
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


def get_curated_smoke_problems() -> List[EvalProblem]:
    """Provides standard HumanEval-compatible problems for offline test validation without network downloads."""
    return [
        EvalProblem(
            task_id="HumanEval/0",
            prompt="def has_close_elements(numbers: list, threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. \"\"\"\n",
            canonical_solution="    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2 and abs(elem - elem2) < threshold:\n                return True\n    return False\n",
            test="def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n",
            entry_point="has_close_elements",
        ),
        EvalProblem(
            task_id="HumanEval/1",
            prompt="def separate_paren_groups(paren_string: str) -> list:\n    \"\"\" Separate groups of nested parentheses. \"\"\"\n",
            canonical_solution="    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string.replace(' ', ''):\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result\n",
            test="def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((()))') == ['()', '(())', '((()))']\n",
            entry_point="separate_paren_groups",
        ),
        EvalProblem(
            task_id="HumanEval/2",
            prompt="def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into integer and decimals. Return decimal part. \"\"\"\n",
            canonical_solution="    return round(number % 1.0, 6)\n",
            test="def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-4\n    assert abs(candidate(123.456) - 0.456) < 1e-4\n",
            entry_point="truncate_number",
        ),
    ]

