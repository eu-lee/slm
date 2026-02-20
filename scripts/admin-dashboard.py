#!/usr/bin/env python3
"""Interactive Admin TUI dashboard for SLM. Requires: pip install textual httpx"""

import argparse
import os
import sys

import httpx
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
)


# ── API helpers ──────────────────────────────────────────────────────────────

def make_client(url: str, key: str) -> httpx.Client:
    return httpx.Client(
        base_url=url.rstrip("/"),
        headers={"x-admin-key": key},
        timeout=15,
    )


def fetch(client: httpx.Client, path: str):
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json()


def fmt_dt(iso) -> str:
    if not iso:
        return "-"
    return iso.replace("T", " ")[:19]


def truncate(text: str, length: int = 80) -> str:
    if not text:
        return "-"
    text = text.replace("\n", " ")
    return text[:length] + "…" if len(text) > length else text


# ── Conversation Detail Screen ───────────────────────────────────────────────

class ConversationDetailScreen(ModalScreen):
    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
    ]

    def __init__(self, client: httpx.Client, conversation: dict) -> None:
        super().__init__()
        self.client = client
        self.conversation = conversation

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            c = self.conversation
            yield Static(
                f"[b]Conversation #{c.get('id', '?')}[/b]  —  "
                f"[cyan]{c.get('title') or 'Untitled'}[/cyan]\n"
                f"User: [green]{c.get('username', '?')}[/green]   "
                f"Messages: {c.get('message_count', '?')}   "
                f"Created: [dim]{fmt_dt(c.get('created_at'))}[/dim]   "
                f"Updated: [dim]{fmt_dt(c.get('updated_at'))}[/dim]",
                id="convo-header",
            )
            yield DataTable(id="messages-table")
        yield Footer()

    def on_mount(self) -> None:
        self.load_messages()

    @work(thread=True)
    def load_messages(self) -> None:
        cid = self.conversation["id"]
        messages = fetch(self.client, f"/api/admin/conversations/{cid}/messages")
        self.app.call_from_thread(self._populate_messages, messages)

    def _populate_messages(self, messages: list) -> None:
        table = self.query_one("#messages-table", DataTable)
        table.add_columns("#", "Role", "Content", "Timestamp")
        for i, m in enumerate(messages, 1):
            table.add_row(
                str(i),
                m["role"],
                truncate(m["content"], 100),
                fmt_dt(m["created_at"]),
            )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


# ── User Detail Screen ───────────────────────────────────────────────────────

class UserDetailScreen(ModalScreen):
    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("enter", "select_conversation", "Open Conversation"),
    ]

    def __init__(self, client: httpx.Client, user: dict) -> None:
        super().__init__()
        self.client = client
        self.user = user
        self._conversations: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            u = self.user
            yield Static(
                f"[b]User: [green]{u['username']}[/green][/b]   ID: {u['id']}\n"
                f"Joined: [cyan]{fmt_dt(u.get('created_at'))}[/cyan]   "
                f"Last active: [yellow]{fmt_dt(u.get('last_active'))}[/yellow]   "
                f"Conversations: {u.get('conversations_count', 0)}   "
                f"Messages: {u.get('messages_count', 0)}",
                id="user-header",
            )
            yield Label("[dim]Conversations:[/dim]")
            yield DataTable(id="user-convos-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.load_conversations()

    @work(thread=True)
    def load_conversations(self) -> None:
        uid = self.user["id"]
        convos = fetch(self.client, f"/api/admin/users/{uid}/conversations")
        self.app.call_from_thread(self._populate_conversations, convos)

    def _populate_conversations(self, convos: list) -> None:
        self._conversations = convos
        table = self.query_one("#user-convos-table", DataTable)
        table.add_columns("ID", "Title", "Messages", "Created", "Updated")
        for c in convos:
            table.add_row(
                str(c["id"]),
                truncate(c["title"] or "Untitled", 50),
                str(c["message_count"]),
                fmt_dt(c["created_at"]),
                fmt_dt(c["updated_at"]),
            )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_select_conversation(self) -> None:
        table = self.query_one("#user-convos-table", DataTable)
        if not self._conversations or table.row_count == 0:
            return
        row_idx = table.cursor_row
        convo = self._conversations[row_idx]
        convo["username"] = self.user["username"]
        self.app.push_screen(ConversationDetailScreen(self.client, convo))


# ── Main App ─────────────────────────────────────────────────────────────────

class AdminDashboard(App):
    CSS = """
    #overview-stats {
        padding: 1 2;
    }
    #convo-header, #user-header {
        padding: 1 2;
        background: $surface;
    }
    DataTable {
        height: 1fr;
    }
    """

    TITLE = "SLM Admin Dashboard"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "select_row", "Open", show=False),
    ]

    def __init__(self, api_url: str, admin_key: str) -> None:
        super().__init__()
        self.client = make_client(api_url, admin_key)
        self._users: list[dict] = []
        self._conversations: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Overview", "Users", "Conversations"):
            with TabPane("Overview", id="tab-overview"):
                yield Static(id="overview-stats")
                yield DataTable(id="top-users-table")
            with TabPane("Users", id="tab-users"):
                yield DataTable(id="users-table", cursor_type="row")
            with TabPane("Conversations", id="tab-conversations"):
                yield DataTable(id="convos-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.load_all_data()

    def action_refresh(self) -> None:
        self.load_all_data()

    @work(thread=True)
    def load_all_data(self) -> None:
        try:
            stats = fetch(self.client, "/api/admin/stats")
            users = fetch(self.client, "/api/admin/users")
            conversations = fetch(self.client, "/api/admin/conversations")
        except httpx.HTTPStatusError as e:
            self.app.call_from_thread(
                self.notify, f"API error: {e.response.status_code}", severity="error"
            )
            return
        except httpx.ConnectError:
            self.app.call_from_thread(
                self.notify, "Could not connect to API", severity="error"
            )
            return
        self.app.call_from_thread(self._populate, stats, users, conversations)

    def _populate(self, stats: dict, users: list, conversations: list) -> None:
        self._users = users
        self._conversations = conversations

        # Overview
        overview = self.query_one("#overview-stats", Static)
        overview.update(
            f"[b cyan]Users:[/b cyan] {stats['total_users']}    "
            f"[b cyan]Conversations:[/b cyan] {stats['total_conversations']}    "
            f"[b cyan]Messages:[/b cyan] {stats['total_messages']}    "
            f"[b cyan]Generations:[/b cyan] {stats['total_generations']}"
        )

        # Top users
        top_table = self.query_one("#top-users-table", DataTable)
        top_table.clear(columns=True)
        top_table.add_columns("#", "Username", "Generations")
        for i, u in enumerate(stats["top_users"], 1):
            top_table.add_row(str(i), u["username"], str(u["generations"]))

        # Users table
        users_table = self.query_one("#users-table", DataTable)
        users_table.clear(columns=True)
        users_table.add_columns("ID", "Username", "Convos", "Messages", "Joined", "Last Active")
        for u in users:
            users_table.add_row(
                str(u["id"]),
                u["username"],
                str(u["conversations_count"]),
                str(u["messages_count"]),
                fmt_dt(u["created_at"]),
                fmt_dt(u["last_active"]),
            )

        # Conversations table
        convos_table = self.query_one("#convos-table", DataTable)
        convos_table.clear(columns=True)
        convos_table.add_columns("ID", "User", "Title", "Messages", "Created", "Updated")
        for c in conversations:
            convos_table.add_row(
                str(c["id"]),
                c["username"],
                truncate(c["title"] or "Untitled", 40),
                str(c["message_count"]),
                fmt_dt(c["created_at"]),
                fmt_dt(c["updated_at"]),
            )

    def action_select_row(self) -> None:
        # Determine which tab is active
        tabbed = self.query_one(TabbedContent)
        active_tab = tabbed.active

        if active_tab == "tab-users":
            table = self.query_one("#users-table", DataTable)
            if not self._users or table.row_count == 0:
                return
            user = self._users[table.cursor_row]
            self.push_screen(UserDetailScreen(self.client, user))

        elif active_tab == "tab-conversations":
            table = self.query_one("#convos-table", DataTable)
            if not self._conversations or table.row_count == 0:
                return
            convo = self._conversations[table.cursor_row]
            self.push_screen(ConversationDetailScreen(self.client, convo))


def main():
    parser = argparse.ArgumentParser(description="SLM Admin Dashboard (TUI)")
    parser.add_argument("--url", default=os.environ.get("SLM_API_URL", "http://localhost:8000"))
    parser.add_argument("--key", default=os.environ.get("SLM_ADMIN_KEY", ""))
    args = parser.parse_args()

    if not args.key:
        print("Error: admin key required (--key or SLM_ADMIN_KEY env var)")
        sys.exit(1)

    app = AdminDashboard(api_url=args.url, admin_key=args.key)
    app.run()


if __name__ == "__main__":
    main()
