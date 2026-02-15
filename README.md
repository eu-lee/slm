# SLM (Small Language Model)

A ~36M parameter GPT chatbot trained on the [TinyChat dataset](https://huggingface.co/datasets/starhopp3r/TinyChat), deployed as a full-stack ChatGPT-style web app.

---

## Architecture

```
            Internet
               |
         [EC2 t2.micro]
               |
    [FastAPI :8000]  <-->  [PostgreSQL (RDS)]
         |
    [PyTorch Model in memory ~140MB]
         |
    [Serves static Next.js build via /]
```

FastAPI serves both the API (`/api/*`) and the static frontend build. No Nginx needed (for now).

**Stack:** FastAPI (Python) + Next.js 14 (TypeScript/Tailwind) + PostgreSQL, all Dockerized.

---

## Project Structure

```
slm/
  model/                       # Model code + training
    loader.py                  # Model architecture + generation logic
    train.py                   # Training script
    token_generator.py         # Tokenizer training
    checkpoints/               # Model weights (~172MB)
    tokenizer/                 # Custom BPE tokenizer (16,387 tokens)

  backend/                     # FastAPI server
    main.py                    # App entrypoint, startup, static file serving
    config.py                  # Path constants and defaults
    database.py                # SQLAlchemy async engine + session
    models.py                  # ORM models (Conversation, Message)
    schemas.py                 # Pydantic request/response schemas
    requirements.txt           # Python dependencies
    inference/
      engine.py                # ModelEngine - loads model, streaming generation
    routers/
      chat.py                  # POST /api/chat (SSE streaming)
      conversations.py         # Conversation CRUD endpoints
  frontend/                    # Next.js 14 app
    app/
      layout.tsx               # Root layout (dark theme)
      page.tsx                 # Chat UI with sidebar
    out/                       # Static export (built, served by FastAPI)

  archive/                     # Old experimental files
    bigram.py                  # Bigram baseline model
    new.py                     # Transformer baseline on Shakespeare
    data/
      shakespeare.txt          # Shakespeare training data

  deploy/                      # (Phase 4) Docker + EC2 scripts
```

---

## How It Works

### The Model
- 6-layer transformer with 8 attention heads, 512 embedding dim
- 384-token context window (this is small — conversation memory is limited)
- Custom tokenizer with special tokens: `<|user|>`, `<|assistant|>`, `<|eos|>`, `<|pad|>`
- Runs on CPU (small enough for free tier), also supports MPS/CUDA

### The Backend
- FastAPI loads the model into memory at startup (~140MB)
- Chat endpoint streams tokens via **Server-Sent Events (SSE)** — you see words appear one by one, like ChatGPT
- Conversations and messages are saved to the database
- When the conversation history exceeds the 384-token context window, it's cleared and the user sees a warning

### The Frontend
- Next.js app built as a static export (just HTML/JS/CSS files)
- Sidebar with conversation list (create, rename, delete)
- Chat area with message bubbles and real-time streaming
- FastAPI serves these static files directly — no separate web server needed

---

## What Each Tool Does

### SQLAlchemy
The Python library that talks to the database. Instead of writing raw SQL like `INSERT INTO messages (role, content) VALUES ('user', 'hello')`, you write Python:
```python
msg = Message(role="user", content="hello")
db.add(msg)
await db.commit()
```
It works with SQLite, PostgreSQL, MySQL, etc. — you swap the database by changing one connection string.

### SQLite vs PostgreSQL
Right now we use **SQLite** (a simple file-based database, `slm_chat.db`). This is perfect for local development — zero setup.

For production on EC2, we'll switch to **PostgreSQL** (via AWS RDS or a Docker container). The switch is literally one line — changing the database URL from:
```
sqlite+aiosqlite:///./slm_chat.db
```
to:
```
postgresql+asyncpg://user:pass@rds-hostname:5432/slm_chat
```

### Docker
Packages everything (Python, dependencies, model weights, frontend build) into a single container that runs identically everywhere. "Works on my machine" → "works everywhere."

---

## Phased Rollout

### Phase 1: Core Chat — DONE
- Model inference engine with streaming
- FastAPI server with SSE chat endpoint
- Next.js chat UI with message bubbles
- Context window warning when history is cleared

### Phase 2: Persistence — DONE
- SQLAlchemy async with SQLite
- Conversations + Messages tables
- CRUD endpoints for conversations
- Sidebar with conversation list (create/rename/delete)
- Chat messages saved to and loaded from database

### Phase 3: Auth (next)
- Users table + registration/login
- JWT tokens + bcrypt password hashing
- Protected routes — conversations are per-user
- Login/register pages on the frontend

### Phase 4: Deploy
- Dockerfile (backend + static frontend in one image)
- docker-compose.yml (backend + PostgreSQL)
- EC2 t2.micro provisioning script
- Open port 8000, deploy, access via public IP

### Phase 5: Polish (optional)
- Nginx reverse proxy for port 80
- HTTPS via Let's Encrypt
- Rate limiting
- Mobile responsive tweaks

---

## Running Locally

**Backend** (terminal 1):
```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend dev server** (terminal 2, for hot reload during development):
```bash
cd frontend
npm run dev
```

**Or production mode** (single terminal — FastAPI serves the built frontend):
```bash
cd frontend && npm run build && cd ..
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/conversations` | List all conversations |
| POST | `/api/conversations` | Create conversation |
| GET | `/api/conversations/{id}` | Get conversation + messages |
| PATCH | `/api/conversations/{id}` | Rename conversation |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| POST | `/api/chat` | Send message (SSE streaming response) |
