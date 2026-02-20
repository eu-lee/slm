#!/usr/bin/env python3
"""Admin TUI dashboard for SLM. Requires: pip install rich httpx"""

import argparse
import os
import sys

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text


def fetch(client: httpx.Client, url: str, path: str):
    resp = client.get(f"{url.rstrip('/')}{path}")
    resp.raise_for_status()
    return resp.json()


def fmt_dt(iso) -> str:
    if not iso:
        return "-"
    return iso.replace("T", " ")[:19]


def main():
    parser = argparse.ArgumentParser(description="SLM Admin Dashboard")
    parser.add_argument("--url", default=os.environ.get("SLM_API_URL", "http://localhost:8000"))
    parser.add_argument("--key", default=os.environ.get("SLM_ADMIN_KEY", ""))
    args = parser.parse_args()

    if not args.key:
        print("Error: admin key required (--key or SLM_ADMIN_KEY env var)")
        sys.exit(1)

    console = Console()
    client = httpx.Client(headers={"x-admin-key": args.key}, timeout=15)

    try:
        stats = fetch(client, args.url, "/api/admin/stats")
        users = fetch(client, args.url, "/api/admin/users")
        conversations = fetch(client, args.url, "/api/admin/conversations")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]API error: {e.response.status_code} {e.response.text}[/red]")
        sys.exit(1)
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to {args.url}[/red]")
        sys.exit(1)

    # --- Overview Panel ---
    overview = Text()
    overview.append(f"  Users: {stats['total_users']}\n", style="bold cyan")
    overview.append(f"  Conversations: {stats['total_conversations']}\n", style="bold cyan")
    overview.append(f"  Messages: {stats['total_messages']}\n", style="bold cyan")
    overview.append(f"  Generations: {stats['total_generations']}", style="bold cyan")

    # --- Top Users Table ---
    top_table = Table(title="Top Users by Generations", show_lines=False)
    top_table.add_column("#", style="dim", width=4)
    top_table.add_column("Username", style="green")
    top_table.add_column("Generations", justify="right", style="magenta")
    for i, u in enumerate(stats["top_users"], 1):
        top_table.add_row(str(i), u["username"], str(u["generations"]))

    console.print()
    console.print(Panel(overview, title="SLM Overview", border_style="blue", expand=False))
    console.print()
    console.print(Columns([top_table], align="center"))

    # --- All Users Table ---
    users_table = Table(title="All Users", show_lines=True)
    users_table.add_column("ID", style="dim", justify="right")
    users_table.add_column("Username", style="green")
    users_table.add_column("Conversations", justify="right")
    users_table.add_column("Messages", justify="right")
    users_table.add_column("Joined", style="cyan")
    users_table.add_column("Last Active", style="yellow")
    for u in users:
        users_table.add_row(
            str(u["id"]),
            u["username"],
            str(u["conversations_count"]),
            str(u["messages_count"]),
            fmt_dt(u["created_at"]),
            fmt_dt(u["last_active"]),
        )

    console.print()
    console.print(users_table)

    # --- Recent Conversations Table ---
    convos_table = Table(title="Recent Conversations", show_lines=True)
    convos_table.add_column("ID", style="dim", justify="right")
    convos_table.add_column("User", style="green")
    convos_table.add_column("Title", max_width=40)
    convos_table.add_column("Messages", justify="right")
    convos_table.add_column("Created", style="cyan")
    convos_table.add_column("Updated", style="yellow")
    for c in conversations:
        convos_table.add_row(
            str(c["id"]),
            c["username"],
            c["title"] or "-",
            str(c["message_count"]),
            fmt_dt(c["created_at"]),
            fmt_dt(c["updated_at"]),
        )

    console.print()
    console.print(convos_table)
    console.print()


if __name__ == "__main__":
    main()
