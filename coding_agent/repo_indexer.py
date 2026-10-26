"""Repository AST Indexer and Call-Graph Map Generator."""

import ast
import os
from typing import Any, Dict, List, Optional
from coding_agent.sandbox.permission_guard import PermissionGuard


class RepositoryIndexer:
    """Extracts symbols, definitions, and concise code maps across a project directory."""

    def __init__(self, workspace_root: str, max_file_size_kb: int = 500):
        self.guard = PermissionGuard(workspace_root)
        self.workspace_root = self.guard.workspace_root
        self.max_file_size_bytes = max_file_size_kb * 1024

    def index_python_file(self, rel_path: str) -> Dict[str, Any]:
        """Parses a Python file and returns classes, methods, functions, and docstrings."""
        abs_path = self.guard.validate_path(rel_path)
        if not os.path.isfile(abs_path) or os.path.getsize(abs_path) > self.max_file_size_bytes:
            return {"file": rel_path, "symbols": []}

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            tree = ast.parse(content, filename=rel_path)
        except Exception:
            return {"file": rel_path, "symbols": []}

        symbols: List[Dict[str, Any]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            "name": child.name,
                            "line": child.lineno,
                            "args": [arg.arg for arg in child.args.args],
                        })
                symbols.append({
                    "type": "class",
                    "name": node.name,
                    "line": node.lineno,
                    "doc": ast.get_docstring(node),
                    "methods": methods,
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({
                    "type": "function",
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "doc": ast.get_docstring(node),
                })

        return {"file": rel_path, "symbols": symbols}

    def generate_repo_map(self, max_depth: int = 4) -> str:
        """Generates a concise structural repository map suitable for LLM prompt context."""
        map_lines: List[str] = [f"Repository Map for: {os.path.basename(self.workspace_root)}"]

        for root, dirs, files in os.walk(self.workspace_root):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "dist", "build")]

            rel_dir = os.path.relpath(root, self.workspace_root)
            depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
            if depth > max_depth:
                continue

            indent = "  " * depth
            if rel_dir != ".":
                map_lines.append(f"{indent}📂 {os.path.basename(root)}/")

            for file in files:
                if file.startswith(".") or file.endswith((".pyc", ".bin", ".pt", ".lock")):
                    continue
                file_rel = os.path.relpath(os.path.join(root, file), self.workspace_root)
                map_lines.append(f"{indent}  📄 {file}")

                # If Python, add symbol outline
                if file.endswith(".py"):
                    idx_data = self.index_python_file(file_rel)
                    for sym in idx_data.get("symbols", []):
                        if sym["type"] == "class":
                            method_names = [m["name"] for m in sym.get("methods", [])]
                            map_lines.append(f"{indent}    class {sym['name']} (methods: {', '.join(method_names[:5])})")
                        elif sym["type"] == "function":
                            arg_list = ", ".join(sym.get("args", [])[:4])
                            map_lines.append(f"{indent}    def {sym['name']}({arg_list})")

        return "\n".join(map_lines)
