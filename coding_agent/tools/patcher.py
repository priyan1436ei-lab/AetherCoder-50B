"""Unified diff generation, inspection, and surgical patch applicator."""

import difflib
from typing import Dict, Tuple
from coding_agent.tools.file_ops import SandboxedFileOps


class CodePatcher:
    """Generates clean diffs and safely applies surgical edits to source files."""

    def __init__(self, file_ops: SandboxedFileOps):
        self.file_ops = file_ops

    def generate_diff(self, rel_path: str, new_content: str) -> str:
        """Computes a unified diff between current disk contents and proposed content."""
        try:
            original = self.file_ops.read_file(rel_path)
        except FileNotFoundError:
            original = ""

        original_lines = original.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
        return "\n".join(diff)

    def apply_patch(self, rel_path: str, new_content: str) -> Tuple[bool, str]:
        """Applies proposed file content and returns (success, unified_diff)."""
        diff = self.generate_diff(rel_path, new_content)
        self.file_ops.write_file(rel_path, new_content)
        return True, diff

    def apply_replacement(self, rel_path: str, target: str, replacement: str) -> Tuple[bool, str]:
        """Surgically replaces a unique code segment in a file and returns diff."""
        original = self.file_ops.read_file(rel_path)
        if target not in original:
            raise ValueError(f"Target code segment not found in {rel_path}.")

        count = original.count(target)
        if count > 1:
            raise ValueError(f"Target code segment occurs {count} times in {rel_path}. It must be unique.")

        updated = original.replace(target, replacement, 1)
        return self.apply_patch(rel_path, updated)
