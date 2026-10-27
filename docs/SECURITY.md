# AetherCoder-50B Security & Sandboxing Policy

Security is a foundational requirement for AetherCoder, particularly when executing autonomous agent loops or deploying the inference API.

---

## 1. Autonomous Coding Agent Sandboxing

The autonomous coding agent runs tools within a strict isolation boundary enforced by `PermissionGuard`:

### Filesystem Confinement
- **Path Traversal Defense**: All target filepaths are lexically and structurally validated to ensure they reside strictly within the designated `workspace_root`.
- **Symlink Escape Defense**: Uses `os.path.realpath` to resolve all symbolic links. Any path whose real physical target resolves outside `workspace_root` raises an immediate `PermissionError`.
- **Sensitive File Blocking**: Direct read or write access to `.env`, `.ssh/id_rsa`, Windows `System32`, or Unix `/etc/passwd` is unconditionally blocked.

### Subprocess Command Isolation
- **Binary Allowlist**: Subprocess execution is restricted strictly to approved testing and diagnostic binaries (`python`, `python3`, `pytest`, `ruff`, `flake8`, `mypy`, `npm`, `node`, `cargo`, `git`).
- **Command Injection Prevention**: Individual argument strings are checked for shell metacharacters (`&&`, `||`, `|`, `` ` ``, `$()`).
- **Dangerous Call Blocking**: Inline scripts passed to interpreters (`-c`) cannot call `os.system`, `subprocess`, `socket`, or destructive filesystem routines.
- **Package Installation Gating**: Unapproved package installs (`npm install`, `pip install`, `cargo install`) require explicit user authorization.
- **Environment Variable Sanitization**: Subprocesses receive a stripped environment where sensitive API keys (`*API_KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSW*`, `AWS_*`, `GITHUB_*`) are excised.
- **Execution Timeouts**: All terminal commands enforce hard timeouts (default: 30-60 seconds) to prevent infinite loops.

---

## 2. API Server Security

- **Authentication**: Endpoints require `Authorization: Bearer <API_KEY>` validated using constant-time string comparison (`secrets.compare_digest`) to prevent timing side-channel attacks.
- **No Unintentional Exposure**: Anonymous access is disabled by default unless explicitly opted in via `AETHERCODER_ALLOW_ANONYMOUS=1`.
- **CORS Restraint**: Cross-Origin Resource Sharing is locked down to safe local origins (`http://localhost:8000`, etc.) by default rather than permitting universal `*` origins.
- **Token Bucket Rate Limiting**: Enforces requests-per-minute limits per client IP with periodic garbage collection to prevent memory exhaustion.
- **Context Limit Safeguards**: Requests requesting token sequences larger than the model's context window are rejected with HTTP 400.

---

## 3. Reporting Vulnerabilities

To report a security vulnerability or exploit vector, please contact the maintainers with a reproducible proof of concept.
