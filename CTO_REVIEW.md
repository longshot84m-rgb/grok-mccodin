# CTO Review — Grok McCodin v0.1.0

## Architecture Decisions

### What we kept from the original plan
- **Typer** for CLI structure — type-safe commands, auto-generated help, shell completion
- **tqdm** for progress indication during API calls
- **Poetry** for packaging — `pip install grok-mccodin` with a global entry point
- **GitHub Actions CI** — lint, type-check, test on every push/PR
- **MIT License**

### What we changed and why

| Original | Changed To | Reason |
|---|---|---|
| Python 3.8 | Python 3.10+ | 3.8 hit EOL Oct 2024. 3.10 gives us `match`, `X \| Y` types, `slots=True` dataclasses |
| `termcolor` | `rich` | Rich gives us markdown rendering, syntax-highlighted diffs, panels, tables — all in one dep |
| `pyautogui` (required) | `pyautogui` (optional group) | GUI automation fails in headless/CI. Made it an optional extra |
| Single `main.py` + `utils.py` | 6-module package | Separation of concerns: client, config, editor, executor, social, utils |
| Hardcoded API keys | `python-dotenv` + `Config` dataclass | Secrets loaded from `.env`, never committed |
| No `.gitignore` | Full `.gitignore` | Prevents committing `__pycache__`, `.env`, `node_modules`, etc. |
| Placeholder tests (`pass`) | 30+ real tests with mocks | Covers config, client, editor, executor, utils |
| No error handling on API calls | `GrokAPIError` exception + graceful fallbacks | CLI doesn't crash on 429/500 |
| No safe delete | Trash-based delete (`.trash/`) | Files moved to `.trash/` before removal — recoverable |

### Module Responsibilities

```
grok_mccodin/
  __init__.py    — version
  __main__.py    — python -m grok_mccodin support
  config.py      — Config dataclass, .env loading
  client.py      — GrokClient (xAI API wrapper)
  editor.py      — parse responses, diffs, file ops
  executor.py    — sandboxed shell/Python execution
  social.py      — X posting, Giphy search
  utils.py       — folder indexing, file reading, logging
  main.py        — Typer app, chat loop, slash commands
```

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Grok executes destructive commands | Safe Lock toggle, blocked-command list, user confirmation |
| API key leakage | `.env` in `.gitignore`, `.env.example` for onboarding |
| File loss from edits/deletes | Diff display before write, trash-based delete |
| Rate limiting (xAI API) | Graceful error handling, no retry loops |
| Large project context overflow | Folder indexer respects `max_depth`, skips binaries |

## Roadmap (next milestones)

### v0.2.0
- [ ] Streaming API responses (SSE) for real-time output
- [ ] Conversation persistence (save/load sessions)
- [ ] `--dry-run` mode (show what would change, don't apply)
- [ ] Richer agent spawning with status tracking

### v0.3.0
- [ ] Plugin system for custom commands
- [ ] Multi-file edit transactions (atomic apply/rollback)
- [ ] Web UI companion (FastAPI + htmx)
- [ ] PyPI auto-publish on tag via GitHub Actions

### v1.0.0
- [ ] Stable API, semver compliance
- [ ] Full documentation site (mkdocs-material)
- [ ] Shell completion install command
- [ ] Docker image for containerized usage

## Quality Gates (CI)

Every push and PR runs:
1. **black** — formatting
2. **ruff** — fast linting
3. **mypy** — type checking
4. **pytest** — test suite

All must pass before merge.
