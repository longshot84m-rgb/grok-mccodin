# Grok McCodin CLI

A Grok-powered CLI for interactive code editing, execution, and X posting.

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
git clone https://github.com/netora/grok-mccodin.git
cd grok-mccodin
pip install poetry
poetry install
poetry run grok-mccodin chat
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
| `/help` | Show available commands |
| `/safelock` | Toggle Safe Lock (blocks execution) |
| `/index` | Re-index the working folder |
| `/screenshot` | Capture a screenshot |
| `/giphy <query>` | Search Giphy for GIFs |
| `/post <text>` | Post to X/Twitter |
| `/run <cmd>` | Run a shell command |
| `/agent <task>` | Spawn a background task |
| `/read <file>` | Read and display a file |
| `/log` | Show the receipt log |
| `/clear` | Clear conversation history |
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
5. **Receipt logging** — every action is logged to `grok_mccodin_log.json`

## Development

```bash
poetry install --with dev

# Run tests
poetry run pytest

# Lint + format
poetry run black .
poetry run ruff check .
poetry run mypy grok_mccodin
```

## License

MIT
