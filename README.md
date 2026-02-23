# Grok McCodin CLI

A Grok-powered CLI for interactive code editing, execution, and X posting — with unlimited conversation memory.

```
   ____           _      __  __       ____          _ _
  / ___|_ __ ___ | | __ |  \/  | ___ / ___|___   __| (_)_ __
 | |  _| '__/ _ \| |/ / | |\/| |/ __| |   / _ \ / _` | | '_ \
 | |_| | | | (_) |   <  | |  | | (__| |__| (_) | (_| | | | | |
  \____|_|  \___/|_|\_\ |_|  |_|\___|\____\___/ \__,_|_|_| |_|
```

## Quick Start

```bash
# Install
pip install grok-mccodin

# Set your API key
export GROK_API_KEY=xai-your-key-here

# Run
grok-mccodin chat
```

## Installation (from source)

```bash
git clone https://github.com/longshot84m-rgb/grok-mccodin.git
cd grok-mccodin
pip install -e ".[dev]"
grok-mccodin chat
```

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GROK_API_KEY` | Yes | xAI API key from https://console.x.ai/ |
| `GROK_MODEL` | No | Model override (default: `grok-3`) |
| `X_API_KEY` | No | X/Twitter OAuth consumer key |
| `X_API_SECRET` | No | X/Twitter OAuth consumer secret |
| `X_ACCESS_TOKEN` | No | X/Twitter OAuth access token |
| `X_ACCESS_SECRET` | No | X/Twitter OAuth access secret |
| `GIPHY_API_KEY` | No | Giphy API key for GIF search |
| `DB_PATH` | No | SQLite database path (default: `project.db`) |
| `GROK_MEMORY_DIR` | No | Session storage directory (default: `~/.grok_mccodin/sessions`) |
| `GROK_TOKEN_BUDGET` | No | Token budget for recent messages before compression (default: `6000`) |
| `GROK_KEEP_RECENT` | No | Number of recent messages always kept uncompressed (default: `10`) |
| `GROK_MEMORY_TOP_K` | No | Number of TF-IDF recalled chunks per query (default: `3`) |

## Usage

### Interactive Chat

```bash
grok-mccodin chat --folder /path/to/project
grok-mccodin chat --verbose
grok-mccodin chat --model grok-3-mini
```

### Slash Commands (inside chat)

| Command | Description |
|---|---|
| `/help` | Show all available commands |
| `/safelock` | Toggle Safe Lock (blocks execution) |
| `/index` | Re-index the working folder |
| `/screenshot` | Capture a screenshot |
| `/giphy <query>` | Search Giphy for GIFs |
| `/post <text>` | Post to X/Twitter |
| `/run <cmd>` | Run a shell command |
| `/agent <task>` | Spawn a background task |
| `/read <file>` | Read and display a file |
| `/search <query>` | Search the web (DuckDuckGo) |
| `/browse <url>` | Fetch and display a web page |
| `/git [cmd]` | Git operations (status/diff/log/commit/branch/push/pull/stash) |
| `/pip <args>` | pip install/list/show/freeze packages |
| `/npm <args>` | npm install/list/run scripts |
| `/sql <query>` | Run a SQL query (SQLite) |
| `/docker [cmd]` | Docker container management (ps/images/logs/stop/build/up/down) |
| `/rag <query>` | Semantic code search (TF-IDF) |
| `/mcp [cmd]` | MCP server management (list/connect/disconnect/tools/call) |
| `/log` | Show the receipt log |
| `/save [name]` | Save conversation session to disk |
| `/load <name>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/memory` | Show memory stats (messages, summaries, token usage) |
| `/clear` | Clear conversation history and memory |
| `/quit` | Exit |

### Standalone Commands

```bash
# Print version
grok-mccodin version

# Index a folder
grok-mccodin index /path/to/project

# Post to X
grok-mccodin post "Hello from Grok McCodin!" --media screenshot.png
```

## How It Works

1. **Chat with Grok** — send natural-language requests to Grok via the xAI API
2. **Auto-apply edits** — Grok responds with fenced code blocks tagged with filenames; the CLI shows a diff and writes the changes
3. **Execute commands** — Grok can include `RUN:` directives that get executed (unless Safe Lock is on)
4. **Create/delete files** — `CREATE:` and `DELETE:` directives are parsed and applied with safety backups
5. **Unlimited memory** — old messages are compressed into summaries, important content is recalled via TF-IDF semantic search, and sessions persist to disk as JSONL
6. **Receipt logging** — every action is logged to `grok_mccodin_log.json`

## Memory System

Grok McCodin uses a rolling compression + semantic recall architecture so conversations never lose context:

- **Recent window** — the last N messages are always sent uncompressed (configurable via `GROK_KEEP_RECENT`)
- **Compression** — when the recent window exceeds the token budget, older messages are compressed into summaries. High-importance messages (code blocks, decisions) are kept verbatim.
- **TF-IDF recall** — when you ask a question, the memory system searches ALL past messages (even compressed ones) for semantically relevant content and injects it into context
- **Session persistence** — sessions auto-save on exit and can be manually saved/loaded with `/save` and `/load`
- **Backup safety** — saving over an existing session creates a `.bak` backup file

## Development

```bash
pip install -e ".[dev]"

# Run tests
pytest -v --tb=short

# Lint + format
ruff check .
ruff format .
mypy grok_mccodin
```

## License

MIT
