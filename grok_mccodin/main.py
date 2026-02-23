"""CLI entry point — Typer app with interactive Grok chat loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from grok_mccodin import __version__
from grok_mccodin.client import GrokClient, GrokAPIError
from grok_mccodin.config import Config
from grok_mccodin.editor import (
    _safe_resolve,
    apply_create,
    apply_delete,
    apply_edit,
    extract_code_blocks,
    extract_commands,
    extract_creates,
    extract_deletes,
)
from grok_mccodin.database import DatabaseError, SQLiteDB
from grok_mccodin.docker import DockerError
from grok_mccodin.docker import summary as docker_summary
from grok_mccodin.executor import _confirm, run_shell, spawn_agent
from grok_mccodin.git import GitError
from grok_mccodin.git import summary as git_summary
from grok_mccodin.mcp import MCPError, MCPRegistry
from grok_mccodin.packages import PackageError, pip_install, npm_install
from grok_mccodin.rag import search_codebase
from grok_mccodin.social import post_to_x, search_giphy
from grok_mccodin.utils import index_folder, log_receipt, read_file_safe, take_screenshot
from grok_mccodin.web import web_fetch, web_search

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
    "/search <query>": "Search the web (DuckDuckGo)",
    "/browse <url>": "Fetch and display a web page",
    "/git [cmd]": "Git operations (status/diff/log/commit/branch)",
    "/pip <args>": "pip install/list/show packages",
    "/npm <args>": "npm install/list/run scripts",
    "/sql <query>": "Run a SQL query (SQLite)",
    "/docker [cmd]": "Docker container management",
    "/rag <query>": "Semantic code search (RAG)",
    "/mcp [cmd]": "MCP server management",
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


# Global MCP registry (lazily initialized)
_mcp_registry: MCPRegistry | None = None


def _get_mcp_registry(folder: Path) -> MCPRegistry:
    """Get or create the MCP registry, loading config from project."""
    global _mcp_registry
    if _mcp_registry is None:
        _mcp_registry = MCPRegistry()
        config_path = folder / "mcp_servers.json"
        _mcp_registry.load_config(config_path)
    return _mcp_registry


def _handle_mcp(arg: str, config: Config, folder: Path) -> None:
    """Handle /mcp subcommands."""
    registry = _get_mcp_registry(folder)

    if not arg or arg == "list":
        names = registry.server_names
        if names:
            console.print("[bold]Configured MCP servers:[/bold]")
            for name in names:
                client = registry.get_client(name)
                status = "[green]connected[/green]" if client else "[dim]disconnected[/dim]"
                console.print(f"  {name}: {status}")
        else:
            console.print(
                "[yellow]No MCP servers configured. "
                "Create mcp_servers.json in your project folder.[/yellow]"
            )
        return None

    sub_parts = arg.split(maxsplit=1)
    sub_cmd = sub_parts[0]
    sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

    try:
        if sub_cmd == "connect" and sub_arg:
            client = registry.connect(sub_arg)
            console.print(f"[green]Connected to MCP server: {sub_arg}[/green]")
            tools = client.list_tools()
            if tools:
                console.print(f"  Available tools: {len(tools)}")
                for t in tools:
                    console.print(f"    - {t.get('name', '?')}: {t.get('description', '')[:80]}")
        elif sub_cmd == "disconnect" and sub_arg:
            registry.disconnect(sub_arg)
            console.print(f"[dim]Disconnected: {sub_arg}[/dim]")
        elif sub_cmd == "tools":
            all_tools = registry.list_all_tools()
            if all_tools:
                for server, tools in all_tools.items():
                    console.print(f"\n[bold]{server}[/bold] ({len(tools)} tools):")
                    for t in tools:
                        console.print(f"  - {t.get('name', '?')}: {t.get('description', '')[:80]}")
            else:
                console.print("[yellow]No connected servers with tools.[/yellow]")
        elif sub_cmd == "call" and sub_arg:
            # /mcp call server_name.tool_name {"arg": "value"}
            call_parts = sub_arg.split(maxsplit=1)
            tool_ref = call_parts[0]
            tool_args_str = call_parts[1] if len(call_parts) > 1 else "{}"
            if "." not in tool_ref:
                console.print("[red]Usage: /mcp call <server>.<tool> [json_args][/red]")
                return None
            server_name, tool_name = tool_ref.split(".", 1)
            client = registry.get_client(server_name)
            if not client:
                console.print(
                    f"[red]Server not connected: {server_name}. Use /mcp connect first.[/red]"
                )
                return None
            import json

            tool_args = json.loads(tool_args_str)
            result = client.call_tool(tool_name, tool_args)
            for block in result:
                if block.get("type") == "text":
                    console.print(block.get("text", ""))
                else:
                    console.print(str(block))
        else:
            console.print(
                "[red]Usage: /mcp list | connect <name> | disconnect <name> | "
                "tools | call <server>.<tool> [json_args][/red]"
            )
    except MCPError as exc:
        console.print(f"[red]MCP error: {exc}[/red]")
    except (ValueError, KeyError) as exc:
        console.print(f"[red]MCP argument error: {exc}[/red]")

    return None


# ---------------------------------------------------------------------------
# Slash command handlers — each takes (arg, config, folder) and returns
# str | None.  Special return values: "__CLEAR__", "__QUIT__".
# ---------------------------------------------------------------------------


def _cmd_help(arg: str, config: Config, folder: Path) -> str | None:
    _print_help()
    return None


def _cmd_safelock(arg: str, config: Config, folder: Path) -> str | None:
    config.safe_lock = not config.safe_lock
    state = "ON" if config.safe_lock else "OFF"
    console.print(f"[bold]Safe Lock: {state}[/bold]")
    return None


def _cmd_index(arg: str, config: Config, folder: Path) -> str | None:
    idx = index_folder(folder)
    console.print(Panel(idx, title="Project Index", border_style="blue"))
    return None


def _cmd_screenshot(arg: str, config: Config, folder: Path) -> str | None:
    path = take_screenshot()
    console.print(f"Screenshot saved: {path}")
    return None


def _cmd_giphy(arg: str, config: Config, folder: Path) -> str | None:
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


def _cmd_post(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /post <tweet text>[/red]")
        return None
    result = post_to_x(arg, config)
    console.print(result)
    log_receipt(config.log_file, action="x_post", detail=result)
    return None


def _cmd_run(arg: str, config: Config, folder: Path) -> str | None:
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


def _cmd_agent(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /agent <task>[/red]")
        return None
    agent_out = spawn_agent(arg, cwd=folder)
    if agent_out["stdout"]:
        console.print(agent_out["stdout"])
    if agent_out["stderr"]:
        console.print(f"[red]{agent_out['stderr']}[/red]")
    return None


def _cmd_read(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /read <filepath>[/red]")
        return None
    resolved = _safe_resolve(arg, folder)
    if resolved is None:
        console.print(f"[red]Blocked: path escapes project folder: {arg}[/red]")
        return None
    content = read_file_safe(resolved)
    console.print(Panel(content, title=arg, border_style="green"))
    return None


def _cmd_search(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /search <query>[/red]")
        return None
    results = web_search(arg)
    if results:
        for i, r in enumerate(results, 1):
            console.print(f"  {i}. [bold]{r['title']}[/bold]")
            console.print(f"     {r['url']}")
            if r.get("snippet"):
                console.print(f"     [dim]{r['snippet'][:120]}[/dim]")
    else:
        console.print("[yellow]No search results found.[/yellow]")
    log_receipt(config.log_file, action="web_search", detail=arg)
    return None


def _cmd_browse(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /browse <url>[/red]")
        return None
    page = web_fetch(arg)
    if page["error"]:
        console.print(f"[red]Error: {page['error']}[/red]")
    else:
        title = page["title"] or page["url"]
        console.print(Panel(page["text"][:3000], title=title, border_style="blue"))
    log_receipt(config.log_file, action="web_browse", detail=arg)
    return None


def _cmd_git(arg: str, config: Config, folder: Path) -> str | None:
    try:
        if not arg:
            console.print(Panel(git_summary(folder), title="Git Summary", border_style="green"))
        else:
            from grok_mccodin import git as git_mod

            sub_parts = arg.split(maxsplit=1)
            sub_cmd = sub_parts[0]
            sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
            if sub_cmd == "status":
                console.print(git_mod.status(folder))
            elif sub_cmd == "diff":
                console.print(git_mod.diff(folder, path=sub_arg))
            elif sub_cmd == "log":
                console.print(git_mod.log(folder))
            elif sub_cmd == "branch":
                console.print(git_mod.branch(folder))
            elif sub_cmd == "add":
                console.print(git_mod.add(sub_arg or ".", cwd=folder))
                console.print("[green]Staged.[/green]")
            elif sub_cmd == "commit":
                if not sub_arg:
                    console.print("[red]Usage: /git commit <message>[/red]")
                else:
                    console.print(git_mod.commit(sub_arg, cwd=folder))
            elif sub_cmd == "push":
                if not _confirm("Push to remote?"):
                    console.print("[dim]Push cancelled.[/dim]")
                    return None
                console.print(git_mod.push(cwd=folder))
            elif sub_cmd == "pull":
                console.print(git_mod.pull(cwd=folder))
            elif sub_cmd == "stash":
                console.print(git_mod.stash(sub_arg or "push", cwd=folder))
            else:
                console.print(f"[red]Unknown git subcommand: {sub_cmd}[/red]")
    except GitError as exc:
        console.print(f"[red]Git error: {exc}[/red]")
    log_receipt(config.log_file, action="git", detail=arg)
    return None


def _cmd_pip(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /pip install <pkg> | /pip list | /pip show <pkg>[/red]")
        return None
    try:
        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0]
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
        if sub_cmd == "install" and sub_arg:
            if not _confirm(f"Install pip package: {sub_arg}?"):
                console.print("[dim]Install cancelled.[/dim]")
                return None
            output = pip_install(sub_arg, cwd=folder)
            console.print(output)
        elif sub_cmd == "list":
            from grok_mccodin.packages import pip_list

            pkgs = pip_list(cwd=folder)
            for pkg in pkgs[:50]:
                console.print(f"  {pkg['name']}=={pkg['version']}")
            if len(pkgs) > 50:
                console.print(f"  ... and {len(pkgs) - 50} more")
        elif sub_cmd == "show" and sub_arg:
            from grok_mccodin.packages import pip_show

            console.print(pip_show(sub_arg, cwd=folder))
        elif sub_cmd == "freeze":
            from grok_mccodin.packages import pip_freeze

            console.print(pip_freeze(cwd=folder))
        else:
            console.print("[red]Usage: /pip install <pkg> | /pip list | /pip show <pkg>[/red]")
    except PackageError as exc:
        console.print(f"[red]pip error: {exc}[/red]")
    log_receipt(config.log_file, action="pip", detail=arg)
    return None


def _cmd_npm(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /npm install [pkg] | /npm list | /npm run <script>[/red]")
        return None
    try:
        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0]
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
        if sub_cmd == "install":
            label = sub_arg if sub_arg else "all dependencies"
            if not _confirm(f"Install npm package: {label}?"):
                console.print("[dim]Install cancelled.[/dim]")
                return None
            output = npm_install(sub_arg or None, cwd=folder)
            console.print(output)
        elif sub_cmd == "list":
            from grok_mccodin.packages import npm_list

            console.print(npm_list(cwd=folder))
        elif sub_cmd == "run" and sub_arg:
            from grok_mccodin.packages import npm_run

            console.print(npm_run(sub_arg, cwd=folder))
        else:
            console.print("[red]Usage: /npm install [pkg] | /npm list | /npm run <script>[/red]")
    except PackageError as exc:
        console.print(f"[red]npm error: {exc}[/red]")
    log_receipt(config.log_file, action="npm", detail=arg)
    return None


def _cmd_sql(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print(
            "[red]Usage: /sql <query>  (uses project.db by default, "
            "or set DB_PATH in .env)[/red]"
        )
        return None
    db_path = str(folder / config.db_path)
    is_read = (
        arg.strip().upper().startswith("SELECT")
        or arg.strip().upper().startswith("PRAGMA")
        or arg.strip().upper() in ("SCHEMA", "TABLES")
    )
    if is_read and not Path(db_path).is_file():
        console.print(f"[yellow]Database not found: {db_path}[/yellow]")
        return None
    try:
        db = SQLiteDB(db_path)
        if arg.strip().upper().startswith("SELECT") or arg.strip().upper().startswith("PRAGMA"):
            rows = db.query(arg)
            if rows:
                tbl = Table(border_style="blue")
                for col in rows[0]:
                    tbl.add_column(col)
                for row in rows[:100]:
                    tbl.add_row(*(str(v) for v in row.values()))
                console.print(tbl)
            else:
                console.print("[dim]No rows returned.[/dim]")
        elif arg.strip().upper() == "SCHEMA":
            console.print(Panel(db.schema() or "[empty]", title="Schema", border_style="blue"))
        elif arg.strip().upper() == "TABLES":
            for t in db.tables():
                console.print(f"  {t}")
        else:
            if not _confirm(f"Execute SQL write: {arg[:80]}?"):
                console.print("[dim]SQL execution cancelled.[/dim]")
                db.close()
                return None
            affected = db.execute(arg)
            console.print(f"[green]OK, {affected} row(s) affected.[/green]")
        db.close()
    except DatabaseError as exc:
        console.print(f"[red]SQL error: {exc}[/red]")
    log_receipt(config.log_file, action="sql", detail=arg[:200])
    return None


def _cmd_docker(arg: str, config: Config, folder: Path) -> str | None:
    try:
        if not arg:
            console.print(Panel(docker_summary(folder), title="Docker", border_style="cyan"))
        else:
            from grok_mccodin import docker as docker_mod

            sub_parts = arg.split(maxsplit=1)
            sub_cmd = sub_parts[0]
            sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
            if sub_cmd == "ps":
                console.print(docker_mod.ps(all_=bool(sub_arg), cwd=folder))
            elif sub_cmd == "images":
                console.print(docker_mod.images(cwd=folder))
            elif sub_cmd == "logs" and sub_arg:
                console.print(docker_mod.logs(sub_arg, cwd=folder))
            elif sub_cmd == "stop" and sub_arg:
                if not _confirm(f"Stop container: {sub_arg}?"):
                    console.print("[dim]Cancelled.[/dim]")
                    return None
                console.print(docker_mod.stop(sub_arg, cwd=folder))
            elif sub_cmd == "rm" and sub_arg:
                if not _confirm(f"Remove container: {sub_arg}?"):
                    console.print("[dim]Cancelled.[/dim]")
                    return None
                console.print(docker_mod.rm(sub_arg, cwd=folder))
            elif sub_cmd == "build":
                if not _confirm("Build Docker image?"):
                    console.print("[dim]Cancelled.[/dim]")
                    return None
                console.print(docker_mod.build(cwd=folder, tag=sub_arg))
            elif sub_cmd == "up":
                if not _confirm("Run docker compose up?"):
                    console.print("[dim]Cancelled.[/dim]")
                    return None
                console.print(docker_mod.compose_up(cwd=folder))
            elif sub_cmd == "down":
                if not _confirm("Run docker compose down?"):
                    console.print("[dim]Cancelled.[/dim]")
                    return None
                console.print(docker_mod.compose_down(cwd=folder))
            else:
                console.print(f"[red]Unknown docker subcommand: {sub_cmd}[/red]")
    except DockerError as exc:
        console.print(f"[red]Docker error: {exc}[/red]")
    log_receipt(config.log_file, action="docker", detail=arg)
    return None


def _cmd_rag(arg: str, config: Config, folder: Path) -> str | None:
    if not arg:
        console.print("[red]Usage: /rag <search query>[/red]")
        return None
    console.print("[dim]Searching codebase...[/dim]")
    rag_output = search_codebase(folder, arg)
    console.print(Panel(rag_output, title="RAG Search Results", border_style="magenta"))
    log_receipt(config.log_file, action="rag_search", detail=arg)
    return None


def _cmd_mcp(arg: str, config: Config, folder: Path) -> str | None:
    _handle_mcp(arg, config, folder)
    return None


def _cmd_log(arg: str, config: Config, folder: Path) -> str | None:
    log_path = Path(config.log_file)
    if log_path.is_file():
        console.print(log_path.read_text(encoding="utf-8"))
    else:
        console.print("[yellow]No log entries yet.[/yellow]")
    return None


def _cmd_clear(arg: str, config: Config, folder: Path) -> str | None:
    return "__CLEAR__"


def _cmd_quit(arg: str, config: Config, folder: Path) -> str | None:
    return "__QUIT__"


# Dispatch table: command name -> handler function
_SLASH_DISPATCH: dict[str, Any] = {
    "/help": _cmd_help,
    "/safelock": _cmd_safelock,
    "/index": _cmd_index,
    "/screenshot": _cmd_screenshot,
    "/giphy": _cmd_giphy,
    "/post": _cmd_post,
    "/run": _cmd_run,
    "/agent": _cmd_agent,
    "/read": _cmd_read,
    "/search": _cmd_search,
    "/browse": _cmd_browse,
    "/git": _cmd_git,
    "/pip": _cmd_pip,
    "/npm": _cmd_npm,
    "/sql": _cmd_sql,
    "/docker": _cmd_docker,
    "/rag": _cmd_rag,
    "/mcp": _cmd_mcp,
    "/log": _cmd_log,
    "/clear": _cmd_clear,
    "/quit": _cmd_quit,
}


def _handle_slash(
    raw: str,
    config: Config,
    folder: Path,
) -> str | None:
    """Process a slash command. Returns a string to display, or None to skip."""
    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    handler = _SLASH_DISPATCH.get(cmd)
    if handler is not None:
        return handler(arg, config, folder)  # type: ignore[no-any-return]

    console.print(f"[red]Unknown command: {cmd}[/red]  Type /help for options.")
    return None


def _process_response(
    reply: str,
    config: Config,
    folder: Path,
) -> None:
    """Parse Grok's reply, render as Markdown, and apply actions."""
    # Show the reply as rendered markdown
    console.print(Markdown(reply))

    # Apply file/command actions
    _process_actions(reply, config, folder)


def _process_actions(
    reply: str,
    config: Config,
    folder: Path,
) -> None:
    """Apply file edits, creates, deletes, and commands from Grok's reply.

    Unlike _process_response, this does NOT render the reply text
    (used when streaming has already printed it).
    """
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
            # Stream response tokens in real time
            chunks: list[str] = []
            for token in client.chat_stream(messages):
                console.print(token, end="", highlight=False)
                chunks.append(token)
            console.print()  # Final newline after streaming
            reply = "".join(chunks)
        except GrokAPIError as exc:
            console.print(f"[bold red]API Error:[/bold red] {exc}")
            continue

        if not reply:
            console.print("[dim]Empty response from Grok.[/dim]")
            continue

        # Append to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

        # Apply file actions (text was already printed via streaming)
        _process_actions(reply, config, folder_path)

        # Log the exchange
        log_receipt(config.log_file, action="chat", user_input=user_input, detail=reply[:200])

    # Cleanup MCP servers on exit
    if _mcp_registry is not None:
        _mcp_registry.disconnect_all()


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
