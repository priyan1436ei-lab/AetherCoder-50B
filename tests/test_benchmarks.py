"""Unit tests for the sandboxed HumanEval and benchmark runner."""

import json
import os
import tempfile
from benchmarks.humaneval_runner import HumanEvalRunner, get_curated_smoke_problems


def test_humaneval_canonical_solution_passes():
    runner = HumanEvalRunner(timeout_seconds=5)
    problems = get_curated_smoke_problems()

    # Create completions dictionary using the canonical solutions
    completions = {p.task_id: p.canonical_solution for p in problems}

    results = runner.evaluate_suite(problems, completions)
    assert results["total_problems"] == len(problems)
    assert results["passed"] == len(problems)
    assert results["pass_at_1"] == 1.0


def test_humaneval_failing_and_timeout_solution():
    runner = HumanEvalRunner(timeout_seconds=2)
    problems = get_curated_smoke_problems()[:2]

    completions = {
        problems[0].task_id: "    return False\n",  # Wrong output
        problems[1].task_id: "    import time; time.sleep(5)\n    return []\n",  # Timeout
    }

    results = runner.evaluate_suite(problems, completions)
    assert results["total_problems"] == 2
    assert results["passed"] == 0
    assert results["failed"] == 2
    assert results["pass_at_1"] == 0.0


def test_humaneval_save_report():
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "benchmark_report.json")
        runner = HumanEvalRunner(timeout_seconds=5)
        problems = get_curated_smoke_problems()[:1]
        completions = {problems[0].task_id: problems[0].canonical_solution}

        report = runner.evaluate_and_save_report(
            problems=problems,
            completions=completions,
            output_json_path=report_path,
            model_name="codex-150m-test",
            notes="Smoke test run",
        )

        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["benchmark"] == "HumanEval-Compatible"
        assert data["model_name"] == "codex-150m-test"
        assert data["summary"]["passed"] == 1
        assert data["summary"]["pass_at_1"] == 1.0
