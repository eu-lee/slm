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


# ── Sort definitions ─────────────────────────────────────────────────────────

# (display_label, dict_key, descending)
OVERVIEW_SORTS = [
    ("Messages", "messages_count", True),
    ("Conversations", "conversations_count", True),
    ("Generations", "generations_count", True),
    ("Date joined (newest)", "created_at", True),
    ("Date joined (oldest)", "created_at", False),
    ("Last active (newest)", "last_active", True),
    ("Last active (oldest)", "last_active", False),
    ("ID", "id", False),
]

USERS_SORTS = [
    ("Messages sent", "messages_count", True),
    ("Conversations", "conversations_count", True),
    ("Date joined (newest)", "created_at", True),
    ("Date joined (oldest)", "created_at", False),
    ("Last active (newest)", "last_active", True),
    ("Last active (oldest)", "last_active", False),
]

CONVO_SORTS = [
    ("Messages", "message_count", True),
    ("Created (newest)", "created_at", True),
    ("Created (oldest)", "created_at", False),
    ("Updated (newest)", "updated_at", True),
    ("Updated (oldest)", "updated_at", False),
]


# ── Conversation Detail Screen ───────────────────────────────────────────────

class ConversationDetailScreen(ModalScreen):
    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
    ]

    def __init__(self, client: httpx.Client, conversation: dict, username: str) -> None:
        super().__init__()
        self.client = client
        self.conversation = conversation
        self.username = username

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            c = self.conversation
            yield Static(
                f"[b]Conversation #{c.get('id', '?')}[/b]  —  "
                f"[cyan]{c.get('title') or 'Untitled'}[/cyan]\n"
                f"User: [green]{self.username}[/green]   "
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
        Binding("s", "cycle_sort", "Sort"),
    ]

    def __init__(self, client: httpx.Client, user: dict) -> None:
        super().__init__()
        self.client = client
        self.user = user
        self._conversations: list[dict] = []
        self._sort_index: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            u = self.user
            yield Static(
                f"[b]User: [green]{u['username']}[/green][/b]   ID: {u['id']}\n"
                f"Joined: [cyan]{fmt_dt(u.get('created_at'))}[/cyan]   "
                f"Last active: [yellow]{fmt_dt(u.get('last_active'))}[/yellow]   "
                f"Conversations: {u.get('conversations_count', 0)}   "
                f"Messages: {u.get('messages_count', 0)}   "
                f"Generations: {u.get('generations_count', 0)}",
                id="user-header",
            )
            yield Static("", id="convo-sort-label")
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
        self._render_conversations_table()

    def _render_conversations_table(self) -> None:
        label_name, sort_key, descending = CONVO_SORTS[self._sort_index]
        label = self.query_one("#convo-sort-label", Static)
        label.update(
            f"[dim]Conversations sorted by:[/dim] [b]{label_name}[/b]  "
            f"[dim](press [b]s[/b] to cycle)[/dim]"
        )

        sorted_convos = sorted(
            self._conversations,
            key=lambda c: c.get(sort_key) or "",
            reverse=descending,
        )

        table = self.query_one("#user-convos-table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Title", "Messages", "Created", "Updated")
        for c in sorted_convos:
            table.add_row(
                str(c["id"]),
                truncate(c["title"] or "Untitled", 50),
                str(c["message_count"]),
                fmt_dt(c["created_at"]),
                fmt_dt(c["updated_at"]),
            )

    def action_cycle_sort(self) -> None:
        if not self._conversations:
            return
        self._sort_index = (self._sort_index + 1) % len(CONVO_SORTS)
        self._render_conversations_table()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not self._conversations:
            return
        _label, sort_key, descending = CONVO_SORTS[self._sort_index]
        sorted_convos = sorted(
            self._conversations,
            key=lambda c: c.get(sort_key) or "",
            reverse=descending,
        )
        convo = sorted_convos[event.cursor_row]
        self.app.push_screen(
            ConversationDetailScreen(self.client, convo, self.user["username"])
        )


# ── Main App ─────────────────────────────────────────────────────────────────

class AdminDashboard(App):
    CSS = """
    #overview-stats {
        padding: 1 2;
    }
    #sort-label, #users-sort-label, #convo-sort-label {
        padding: 0 2;
        color: $text-muted;
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
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "cycle_sort", "Sort"),
    ]

    def __init__(self, api_url: str, admin_key: str) -> None:
        super().__init__()
        self.client = make_client(api_url, admin_key)
        self._users: list[dict] = []
        self._overview_sort_index: int = 0
        self._users_sort_index: int = 0
        # Maps table widget id → (sort_list, sort_index_attr)
        self._table_sort_map = {
            "overview-table": (OVERVIEW_SORTS, "_overview_sort_index"),
            "users-table": (USERS_SORTS, "_users_sort_index"),
        }

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Overview", "Users"):
            with TabPane("Overview", id="tab-overview"):
                yield Static(id="overview-stats")
                yield Static("", id="sort-label")
                yield DataTable(id="overview-table", cursor_type="row")
            with TabPane("Users", id="tab-users"):
                yield Static("", id="users-sort-label")
                yield DataTable(id="users-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.load_all_data()

    def action_refresh(self) -> None:
        self.load_all_data()

    def action_cycle_sort(self) -> None:
        tabbed = self.query_one(TabbedContent)
        if tabbed.active == "tab-overview":
            self._overview_sort_index = (self._overview_sort_index + 1) % len(OVERVIEW_SORTS)
            self._render_overview_table()
        elif tabbed.active == "tab-users":
            self._users_sort_index = (self._users_sort_index + 1) % len(USERS_SORTS)
            self._render_users_table()

    @work(thread=True)
    def load_all_data(self) -> None:
        try:
            stats = fetch(self.client, "/api/admin/stats")
            users = fetch(self.client, "/api/admin/users")
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
        self.app.call_from_thread(self._populate, stats, users)

    def _populate(self, stats: dict, users: list) -> None:
        self._users = users

        # Overview stats
        overview = self.query_one("#overview-stats", Static)
        overview.update(
            f"[b cyan]Users:[/b cyan] {stats['total_users']}    "
            f"[b cyan]Conversations:[/b cyan] {stats['total_conversations']}    "
            f"[b cyan]Messages:[/b cyan] {stats['total_messages']}    "
            f"[b cyan]Generations:[/b cyan] {stats['total_generations']}"
        )

        self._render_overview_table()
        self._render_users_table()

    def _render_overview_table(self) -> None:
        label_name, sort_key, descending = OVERVIEW_SORTS[self._overview_sort_index]

        label = self.query_one("#sort-label", Static)
        label.update(
            f"[dim]Sorted by:[/dim] [b]{label_name}[/b]  "
            f"[dim](press [b]s[/b] to cycle)[/dim]"
        )

        sorted_users = sorted(
            self._users,
            key=lambda u: u.get(sort_key, 0) or 0,
            reverse=descending,
        )

        table = self.query_one("#overview-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Username", "Messages", "Conversations", "Generations", "Joined", "Last Active")
        for i, u in enumerate(sorted_users, 1):
            table.add_row(
                str(i),
                u["username"],
                str(u["messages_count"]),
                str(u["conversations_count"]),
                str(u["generations_count"]),
                fmt_dt(u["created_at"]),
                fmt_dt(u["last_active"]),
            )

    def _render_users_table(self) -> None:
        label_name, sort_key, descending = USERS_SORTS[self._users_sort_index]

        label = self.query_one("#users-sort-label", Static)
        label.update(
            f"[dim]Sorted by:[/dim] [b]{label_name}[/b]  "
            f"[dim](press [b]s[/b] to cycle)[/dim]"
        )

        sorted_users = sorted(
            self._users,
            key=lambda u: u.get(sort_key) or "",
            reverse=descending,
        )

        table = self.query_one("#users-table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Username", "Conversations", "Messages", "Joined", "Last Active")
        for u in sorted_users:
            table.add_row(
                str(u["id"]),
                u["username"],
                str(u["conversations_count"]),
                str(u["messages_count"]),
                fmt_dt(u["created_at"]),
                fmt_dt(u["last_active"]),
            )

    def _get_sorted_users(self, sort_list, sort_index):
        """Return users sorted by the given sort definition."""
        _label, sort_key, descending = sort_list[sort_index]
        return sorted(
            self._users,
            key=lambda u: u.get(sort_key) or "",
            reverse=descending,
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table_id = event.data_table.id
        if table_id not in self._table_sort_map or not self._users:
            return
        sort_list, sort_index_attr = self._table_sort_map[table_id]
        sort_index = getattr(self, sort_index_attr)
        sorted_users = self._get_sorted_users(sort_list, sort_index)
        user = sorted_users[event.cursor_row]
        self.push_screen(UserDetailScreen(self.client, user))


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
