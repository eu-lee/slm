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
- **Infra:** Docker Compose on EC2, Caddy (auto HTTPS), PostgreSQL, Vercel

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
  frontend/
    app/
      layout.tsx               # Root layout
      page.tsx                 # Chat UI
      login/page.tsx           # Auth page
  deploy/
    Dockerfile                 # Python 3.11 + CPU PyTorch
    docker-compose.yml         # app + db + caddy
    Caddyfile                  # Auto HTTPS reverse proxy
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
