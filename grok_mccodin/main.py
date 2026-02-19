"""CLI entry point — Typer app with interactive Grok chat loop."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from grok_mccodin import __version__
from grok_mccodin.client import GrokClient, GrokAPIError
from grok_mccodin.config import Config
from grok_mccodin.editor import (
    apply_create,
    apply_delete,
    apply_edit,
    extract_code_blocks,
    extract_commands,
    extract_creates,
    extract_deletes,
)
from grok_mccodin.executor import run_shell, spawn_agent
from grok_mccodin.social import post_to_x, search_giphy
from grok_mccodin.utils import index_folder, log_receipt, read_file_safe, take_screenshot

app = typer.Typer(
    name="grok-mccodin",
    help="Grok-powered CLI for code editing, execution, and X posting.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BANNER = r"""
   ____           _      __  __       ____          _ _
  / ___|_ __ ___ | | __ |  \/  | ___ / ___|___   __| (_)_ __
 | |  _| '__/ _ \| |/ / | |\/| |/ __| |   / _ \ / _` | | '_ \
 | |_| | | | (_) |   <  | |  | | (__| |__| (_) | (_| | | | | |
  \____|_|  \___/|_|\_\ |_|  |_|\___|\____\___/ \__,_|_|_| |_|
"""

SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/safelock": "Toggle Safe Lock (blocks execution)",
    "/index": "Re-index the working folder",
    "/screenshot": "Take a screenshot",
    "/giphy <query>": "Search Giphy for a GIF",
    "/post <text>": "Post to X/Twitter",
    "/run <cmd>": "Run a shell command",
    "/agent <task>": "Spawn a background agent task",
    "/read <file>": "Read and display a file",
    "/log": "Show the receipt log",
    "/clear": "Clear conversation history",
    "/quit": "Exit the CLI",
}


def _print_help() -> None:
    table = Table(title="Grok McCodin Commands", border_style="cyan")
    table.add_column("Command", style="bold green")
    table.add_column("Description")
    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)


def _handle_slash(
    raw: str,
    config: Config,
    folder: Path,
) -> str | None:
    """Process a slash command. Returns a string to display, or None to skip."""
    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        _print_help()
        return None

    if cmd == "/safelock":
        config.safe_lock = not config.safe_lock
        state = "ON" if config.safe_lock else "OFF"
        console.print(f"[bold]Safe Lock: {state}[/bold]")
        return None

    if cmd == "/index":
        idx = index_folder(folder)
        console.print(Panel(idx, title="Project Index", border_style="blue"))
        return None

    if cmd == "/screenshot":
        path = take_screenshot()
        console.print(f"Screenshot saved: {path}")
        return None

    if cmd == "/giphy":
        if not arg:
            console.print("[red]Usage: /giphy <search query>[/red]")
            return None
        results = search_giphy(arg, config)
        if results:
            for i, r in enumerate(results, 1):
                console.print(f"  {i}. {r['title']}: {r['url']}")
        else:
            console.print("[yellow]No results found.[/yellow]")
        return None

    if cmd == "/post":
        if not arg:
            console.print("[red]Usage: /post <tweet text>[/red]")
            return None
        result = post_to_x(arg, config)
        console.print(result)
        log_receipt(config.log_file, action="x_post", detail=result)
        return None

    if cmd == "/run":
        if not arg:
            console.print("[red]Usage: /run <command>[/red]")
            return None
        shell_out = run_shell(arg, cwd=folder, safe_lock=config.safe_lock)
        if shell_out["stdout"]:
            console.print(shell_out["stdout"])
        if shell_out["stderr"]:
            console.print(f"[red]{shell_out['stderr']}[/red]")
        log_receipt(config.log_file, action="shell_run", detail=arg)
        return None

    if cmd == "/agent":
        if not arg:
            console.print("[red]Usage: /agent <task>[/red]")
            return None
        agent_out = spawn_agent(arg, cwd=folder)
        if agent_out["stdout"]:
            console.print(agent_out["stdout"])
        if agent_out["stderr"]:
            console.print(f"[red]{agent_out['stderr']}[/red]")
        return None

    if cmd == "/read":
        if not arg:
            console.print("[red]Usage: /read <filepath>[/red]")
            return None
        content = read_file_safe(folder / arg)
        console.print(Panel(content, title=arg, border_style="green"))
        return None

    if cmd == "/log":
        log_path = Path(config.log_file)
        if log_path.is_file():
            console.print(log_path.read_text(encoding="utf-8"))
        else:
            console.print("[yellow]No log entries yet.[/yellow]")
        return None

    if cmd == "/clear":
        return "__CLEAR__"

    if cmd == "/quit":
        return "__QUIT__"

    console.print(f"[red]Unknown command: {cmd}[/red]  Type /help for options.")
    return None


def _process_response(
    reply: str,
    config: Config,
    folder: Path,
) -> None:
    """Parse Grok's reply and apply edits, creates, deletes, commands."""
    # Show the reply as rendered markdown
    console.print(Markdown(reply))

    # Apply file edits (code blocks with filenames)
    for block in extract_code_blocks(reply):
        if block["filename"]:
            if config.safe_lock:
                console.print(f"[yellow][Safe Lock] Skipping edit: {block['filename']}[/yellow]")
                continue
            status = apply_edit(block["filename"], block["code"], base_dir=folder)
            console.print(f"  -> {status}")
            log_receipt(config.log_file, action="edit", detail=block["filename"])

    # Apply creates
    for create in extract_creates(reply):
        if config.safe_lock:
            console.print(f"[yellow][Safe Lock] Skipping create: {create['path']}[/yellow]")
            continue
        status = apply_create(create["path"], create["code"], base_dir=folder)
        console.print(f"  -> {status}")
        log_receipt(config.log_file, action="create", detail=create["path"])

    # Apply deletes
    for delete_path in extract_deletes(reply):
        if config.safe_lock:
            console.print(f"[yellow][Safe Lock] Skipping delete: {delete_path}[/yellow]")
            continue
        status = apply_delete(delete_path, base_dir=folder)
        console.print(f"  -> {status}")
        log_receipt(config.log_file, action="delete", detail=delete_path)

    # Execute RUN commands
    for command in extract_commands(reply):
        if config.safe_lock:
            console.print(f"[yellow][Safe Lock] Skipping run: {command}[/yellow]")
            continue
        console.print(f"\n[cyan]Executing:[/cyan] {command}")
        result = run_shell(command, cwd=folder, safe_lock=config.safe_lock)
        if result["stdout"]:
            console.print(result["stdout"])
        if result["stderr"]:
            console.print(f"[red]{result['stderr']}[/red]")
        log_receipt(config.log_file, action="run", detail=command)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@app.command()
def chat(
    folder: str = typer.Option(".", "--folder", "-f", help="Working directory for the project."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    model: str = typer.Option(
        "", "--model", "-m", help="Override the Grok model (e.g. grok-3-mini)."
    ),
) -> None:
    """Start an interactive Grok McCodin chat session."""
    # Logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Config
    config = Config.from_env()
    if model:
        config.grok_model = model

    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        console.print(f"[red]Folder not found: {folder_path}[/red]")
        raise typer.Exit(1)

    config.working_dir = str(folder_path)

    # Validate API key
    if not config.has_grok_key:
        console.print(
            "[bold red]GROK_API_KEY not set.[/bold red]\n"
            "Create a .env file with:\n"
            "  GROK_API_KEY=xai-your-key-here\n"
            "Or export it: export GROK_API_KEY=xai-..."
        )
        raise typer.Exit(1)

    # Banner
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    console.print(
        f"[dim]v{__version__}  |  model: {config.grok_model}  |  folder: {folder_path}[/dim]"
    )
    console.print("[dim]Type /help for commands, /quit to exit.[/dim]\n")

    # Build context and state
    client = GrokClient(config)
    history: list[dict[str, str]] = []
    max_history = 40  # Keep last N messages to avoid token overflow

    # Main loop
    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            result = _handle_slash(user_input, config, folder_path)
            if result == "__QUIT__":
                console.print("[dim]Goodbye![/dim]")
                break
            if result == "__CLEAR__":
                history.clear()
                console.print("[dim]History cleared.[/dim]")
            continue

        # Re-index folder each turn so Grok sees recent file changes
        folder_index = index_folder(folder_path)

        # Trim history to avoid exceeding model context window
        if len(history) > max_history:
            history = history[-max_history:]

        # Build messages and call Grok
        messages = client.build_messages(history, user_input, context=folder_index)

        try:
            reply = client.chat(messages)
        except GrokAPIError as exc:
            console.print(f"[bold red]API Error:[/bold red] {exc}")
            continue

        # Append to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

        # Process the response (render + apply actions)
        _process_response(reply, config, folder_path)

        # Log the exchange
        log_receipt(config.log_file, action="chat", user_input=user_input, detail=reply[:200])


@app.command()
def version() -> None:
    """Print the version and exit."""
    console.print(f"grok-mccodin v{__version__}")


@app.command()
def index(
    folder: str = typer.Argument(".", help="Folder to index."),
) -> None:
    """Index a project folder and print its structure."""
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        console.print(f"[red]Not a directory: {folder_path}[/red]")
        raise typer.Exit(1)
    idx = index_folder(folder_path)
    console.print(Panel(idx, title=str(folder_path), border_style="blue"))


@app.command()
def post(
    text: str = typer.Argument(..., help="Tweet text to post."),
    media: str = typer.Option(None, "--media", "-m", help="Path to media file to attach."),
) -> None:
    """Post a message to X/Twitter."""
    config = Config.from_env()
    result = post_to_x(text, config, media_path=media)
    console.print(result)


# Allow running directly with `python -m grok_mccodin`
if __name__ == "__main__":
    app()
