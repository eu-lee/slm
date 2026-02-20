# SLM (Small Language Model)

A ~36M parameter GPT chatbot trained on the [TinyChat dataset](https://huggingface.co/datasets/starhopp3r/TinyChat), deployed as a full-stack ChatGPT-style [web app](https://eugpt.chat).

## Architecture

```
Vercel (CDN)                         EC2 Instance
┌───────────────────┐                ┌──────────────────────────────┐
│  Next.js Frontend │  ── HTTPS ──>  │  docker-compose              │
│  eugpt.chat       │                │  ├── caddy (:443)            │
│                   │                │  │   └── reverse proxy + SSL │
└───────────────────┘                │  ├── app (:8000)             │
                                     │  │   ├── FastAPI             │
                                     │  │   ├── PyTorch model (CPU) │
                                     │  │   └── uvicorn             │
                                     │  └── db (:5432)              │
                                     │      └── PostgreSQL + volume │
                                     └──────────────────────────────┘
```

## Stack

- **Model:** 6-layer transformer, 8 attention heads, 512 embedding dim, 16,387 vocab size, 384-token context window
- **Backend:** FastAPI, PyTorch (CPU inference), SQLAlchemy async, JWT auth
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Infra:** Docker Compose on EC2, Caddy (auto HTTPS), PostgreSQL, Vercel, GitHub Actions CI/CD

## Project Structure

```
slm/
  model/                         # Training only (not deployed)
    train.py                     # Training script
    token_generator.py           # BPE tokenizer training
    loader.py                    # Data loading
  models/                        # Inference artifacts (gitignored)
    eugpt-36m/
      checkpoint.pth             # Model weights
      tokenizer/                 # Custom BPE tokenizer (16,387 tokens)
  backend/
    main.py                    # FastAPI entrypoint
    config.py                  # Env-based configuration
    database.py                # SQLAlchemy async engine
    models.py                  # ORM models (User, Conversation, Message)
    auth.py                    # JWT auth + bcrypt
    inference/
      engine.py                # Model loading + streaming generation
    routers/
      auth.py                  # Login / register endpoints
      chat.py                  # SSE streaming chat
      conversations.py         # Conversation CRUD
      admin.py                 # Admin stats + drill-down endpoints
  frontend/
    app/
      layout.tsx               # Root layout
      page.tsx                 # Chat UI
      login/page.tsx           # Auth page
  scripts/
    admin-dashboard.py         # Interactive TUI admin dashboard
  deploy/
    Dockerfile                 # Python 3.11 + CPU PyTorch
    docker-compose.yml         # app + db + caddy
    Caddyfile                  # Auto HTTPS reverse proxy
  .github/workflows/
    deploy.yml                 # Auto-deploy backend to EC2 on push
```

## Running Locally

```bash
# Backend
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm run dev
```

## API

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/health` | No | Health check |
| POST | `/api/auth/register` | No | Create account |
| POST | `/api/auth/login` | No | Get JWT token |
| POST | `/api/chat/guest` | No | Guest chat (SSE) |
| POST | `/api/chat` | JWT | Authenticated chat (SSE) |
| GET | `/api/conversations` | JWT | List conversations |
| POST | `/api/conversations` | JWT | Create conversation |
| GET | `/api/conversations/{id}` | JWT | Get conversation + messages |
| PATCH | `/api/conversations/{id}` | JWT | Rename conversation |
| DELETE | `/api/conversations/{id}` | JWT | Delete conversation |
| GET | `/api/admin/stats` | Admin key | Overview totals + top users |
| GET | `/api/admin/users` | Admin key | All users with per-user stats |
| GET | `/api/admin/users/{id}/conversations` | Admin key | Conversations for a user |
| GET | `/api/admin/conversations/{id}/messages` | Admin key | Messages in a conversation |
| GET | `/api/admin/conversations` | Admin key | Recent conversations |

Admin endpoints require the `X-Admin-Key` header matching the `ADMIN_API_KEY` environment variable.

## Admin Dashboard

An interactive terminal UI for monitoring users, conversations, and generations.

```bash
pip install textual httpx
python scripts/admin-dashboard.py --url http://localhost:8000 --key <admin-key>
```

You can also set environment variables instead of flags: `SLM_API_URL` and `SLM_ADMIN_KEY`.

### Screens

- **Overview** — totals (users, conversations, messages, generations) and a sortable table of all users with messages, conversations, generations, join date, and last active
- **Users** — sortable user listing; press Enter on any user to view their conversation history
- **User Detail** — user info header + sortable list of their conversations; press Enter to view messages
- **Conversation Detail** — conversation info header + full message history

### Key Bindings

| Key | Action |
|-----|--------|
| Tab / Shift+Tab | Switch between tabs |
| Arrow keys | Navigate within tables |
| Enter | Drill into selected row |
| Escape | Go back to previous screen |
| s | Cycle sort order |
| r | Refresh data |
| q | Quit |
