# Security Policy & Sandbox Threat Model

## 1. Threat Model & Confinement Boundary

The Coding LLM ecosystem incorporates multi-layered defense to isolate model code generation and prevent host system compromise:

### Filesystem Boundary Enforcement (`PermissionGuard`)
* **Path Traversal Blocker**: All file reads, writes, and listings resolve via strict `os.path.commonpath` verification against the designated workspace root. Any attempt to access parent directories via `../`, `..\\`, or symlinks throws an immediate `PermissionError`.
* **Sensitive File Pattern Blocker**: Access to sensitive credential and system paths (`.ssh`, `id_rsa`, `.env`, `Windows\System32`, `/etc/passwd`) is denied by regex policy.

### Sandboxed Terminal Execution (`SandboxedTerminal`)
* **Command Allowlist**: Only approved developer utilities are permitted (`python`, `pytest`, `ruff`, `flake8`, `mypy`, `npm`, `node`, `cargo`, `git`).
* **Destructive Command Filters**: High-risk patterns such as `rm -rf /`, `del /s /q`, `format`, `curl | bash`, and system shutdown commands are permanently blocked.
* **Execution Timeouts**: All test runs are enforced with hard subprocess timeouts (default 60s) to guard against infinite loops and resource exhaustion attacks.

### Data Ingestion Sanitization (`SecretScrubber` & `LicenseFilter`)
* **Credential Scrubber**: Training datasets and prompts are automatically scanned using high-confidence regexes and Shannon entropy analysis to redact AWS keys, GitHub PATs, private keys, and database passwords with `<SECRET_REDACTED>`.
* **License Filter**: Ensures only permissively licensed source code (MIT, Apache-2.0, BSD, ISC, CC0) is ingested into the training pipeline.
