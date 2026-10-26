"""Sandboxed file operations (read, write, list) with strict boundary enforcement."""

import os
from typing import List, Optional
from coding_agent.sandbox.permission_guard import PermissionGuard


class SandboxedFileOps:
    """Provides safe filesystem operations bounded to an authorized project root."""

    def __init__(self, workspace_root: str):
        self.guard = PermissionGuard(workspace_root)

    def read_file(self, rel_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Reads a file within the workspace, optionally sliced by line range."""
        abs_path = self.guard.validate_path(rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {rel_path}")

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if start_line is not None or end_line is not None:
            s = (start_line - 1) if start_line is not None and start_line > 0 else 0
            e = end_line if end_line is not None else len(lines)
            return "".join(lines[s:e])

        return "".join(lines)

    def write_file(self, rel_path: str, content: str) -> None:
        """Writes content to a file inside the authorized workspace, creating parent directories."""
        abs_path = self.guard.validate_path(rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

    def list_files(self, rel_dir: str = ".") -> List[str]:
        """Lists all files in a relative directory."""
        abs_dir = self.guard.validate_path(rel_dir)
        if not os.path.isdir(abs_dir):
            raise NotADirectoryError(f"Not a directory: {rel_dir}")

        file_list = []
        for root, _, files in os.walk(abs_dir):
            for file in files:
                rel = os.path.relpath(os.path.join(root, file), self.guard.workspace_root)
                file_list.append(rel)
        return sorted(file_list)
