"""Repository AST Indexer and Call-Graph Map Generator.

Provides AST-level symbol extraction for Python and pattern-based symbol extraction
for TypeScript/JavaScript, Go, Rust, and C/C++, plus cross-file dependency mapping.
"""

import ast
import os
import re
from typing import Any, Dict, List, Optional, Set
from coding_agent.sandbox.permission_guard import PermissionGuard


class RepositoryIndexer:
    """Extracts symbols, definitions, and concise code maps across a project directory."""

    def __init__(self, workspace_root: str, max_file_size_kb: int = 500):
        self.guard = PermissionGuard(workspace_root)
        self.workspace_root = self.guard.workspace_root
        self.max_file_size_bytes = max_file_size_kb * 1024
        self._symbol_cache: Dict[str, List[Dict[str, Any]]] = {}

    def index_python_file(self, rel_path: str) -> Dict[str, Any]:
        """Parses a Python file and returns classes, methods, functions, docstrings, and imports."""
        abs_path = self.guard.validate_path(rel_path)
        if not os.path.isfile(abs_path) or os.path.getsize(abs_path) > self.max_file_size_bytes:
            return {"file": rel_path, "symbols": [], "imports": []}

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            tree = ast.parse(content, filename=rel_path)
        except Exception:
            return {"file": rel_path, "symbols": [], "imports": []}

        symbols: List[Dict[str, Any]] = []
        imports: List[str] = []

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
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return {"file": rel_path, "symbols": symbols, "imports": imports}

    def index_generic_code_file(self, rel_path: str) -> Dict[str, Any]:
        """Extracts key symbols from non-Python code files using language-specific patterns."""
        abs_path = self.guard.validate_path(rel_path)
        if not os.path.isfile(abs_path) or os.path.getsize(abs_path) > self.max_file_size_bytes:
            return {"file": rel_path, "symbols": [], "imports": []}

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            return {"file": rel_path, "symbols": [], "imports": []}

        ext = os.path.splitext(rel_path)[1].lower()
        symbols: List[Dict[str, Any]] = []
        imports: List[str] = []

        # Patterns for TypeScript/JavaScript, Go, Rust, C/C++
        ts_func_pattern = re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_$]+)\s*\(')
        ts_class_pattern = re.compile(r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+([a-zA-Z0-9_$]+)')
        ts_type_pattern = re.compile(r'^\s*(?:export\s+)?(?:interface|type)\s+([a-zA-Z0-9_$]+)')
        ts_import_pattern = re.compile(r'^\s*import\s+.*from\s+[\'"]([^\'"]+)[\'"]')

        go_func_pattern = re.compile(r'^\s*func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\(')
        go_type_pattern = re.compile(r'^\s*type\s+([a-zA-Z0-9_]+)\s+(?:struct|interface)')
        go_import_pattern = re.compile(r'^\s*import\s+(?:\([^)]+\)|"([^"]+)")')

        rust_fn_pattern = re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)\s*(?:<[^>]+>)?\s*\(')
        rust_struct_pattern = re.compile(r'^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([a-zA-Z0-9_]+)')
        rust_use_pattern = re.compile(r'^\s*use\s+([^;]+);')

        cpp_pattern = re.compile(r'^\s*(?:template\s*<[^>]+>\s*)?(?:class|struct)\s+([a-zA-Z0-9_]+)')

        for idx, line in enumerate(lines, start=1):
            if ext in (".ts", ".tsx", ".js", ".jsx"):
                if m := ts_func_pattern.match(line):
                    symbols.append({"type": "function", "name": m.group(1), "line": idx})
                elif m := ts_class_pattern.match(line):
                    symbols.append({"type": "class", "name": m.group(1), "line": idx})
                elif m := ts_type_pattern.match(line):
                    symbols.append({"type": "type", "name": m.group(1), "line": idx})
                elif m := ts_import_pattern.match(line):
                    imports.append(m.group(1))

            elif ext == ".go":
                if m := go_func_pattern.match(line):
                    symbols.append({"type": "function", "name": m.group(1), "line": idx})
                elif m := go_type_pattern.match(line):
                    symbols.append({"type": "struct", "name": m.group(1), "line": idx})
                elif m := go_import_pattern.match(line):
                    if m.group(1):
                        imports.append(m.group(1))

            elif ext == ".rs":
                if m := rust_fn_pattern.match(line):
                    symbols.append({"type": "function", "name": m.group(1), "line": idx})
                elif m := rust_struct_pattern.match(line):
                    symbols.append({"type": "struct", "name": m.group(1), "line": idx})
                elif m := rust_use_pattern.match(line):
                    imports.append(m.group(1).strip())

            elif ext in (".c", ".cpp", ".cc", ".h", ".hpp"):
                if m := cpp_pattern.match(line):
                    symbols.append({"type": "class", "name": m.group(1), "line": idx})

        return {"file": rel_path, "symbols": symbols, "imports": imports}

    def index_file(self, rel_path: str) -> Dict[str, Any]:
        """Indexes any supported code file dynamically."""
        ext = os.path.splitext(rel_path)[1].lower()
        if ext == ".py":
            data = self.index_python_file(rel_path)
        elif ext in (".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".c", ".cpp", ".cc", ".h", ".hpp"):
            data = self.index_generic_code_file(rel_path)
        else:
            data = {"file": rel_path, "symbols": [], "imports": []}
        self._symbol_cache[rel_path] = data.get("symbols", [])
        return data

    def find_symbol_definition(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Searches for a symbol definition across the entire workspace repository."""
        matches: List[Dict[str, Any]] = []
        for root, dirs, files in os.walk(self.workspace_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "dist", "build")]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in (".py", ".ts", ".tsx", ".js", ".go", ".rs", ".cpp", ".c", ".h"):
                    rel_path = os.path.relpath(os.path.join(root, file), self.workspace_root)
                    data = self.index_file(rel_path)
                    for sym in data.get("symbols", []):
                        if sym.get("name") == symbol_name:
                            matches.append({
                                "file": rel_path,
                                "type": sym.get("type"),
                                "line": sym.get("line"),
                                "details": sym,
                            })
        return matches

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

            for file in sorted(files):
                if file.startswith(".") or file.endswith((".pyc", ".bin", ".pt", ".lock")):
                    continue
                file_rel = os.path.relpath(os.path.join(root, file), self.workspace_root)
                map_lines.append(f"{indent}  📄 {file}")

                # Add symbol outline for supported languages
                ext = os.path.splitext(file)[1].lower()
                if ext in (".py", ".ts", ".js", ".go", ".rs", ".cpp", ".c"):
                    idx_data = self.index_file(file_rel)
                    for sym in idx_data.get("symbols", [])[:8]:
                        stype = sym.get("type", "symbol")
                        sname = sym.get("name", "")
                        line_no = sym.get("line", "")
                        if stype == "class":
                            methods = [m["name"] for m in sym.get("methods", [])[:4]]
                            m_str = f" (methods: {', '.join(methods)})" if methods else ""
                            map_lines.append(f"{indent}    class {sname}{m_str} [L{line_no}]")
                        elif stype in ("function", "func"):
                            args = sym.get("args", [])
                            arg_str = f"({', '.join(args[:4])})" if args else "()"
                            map_lines.append(f"{indent}    def {sname}{arg_str} [L{line_no}]")
                        elif stype in ("interface", "type", "struct"):
                            map_lines.append(f"{indent}    {stype} {sname} [L{line_no}]")

        return "\n".join(map_lines)
