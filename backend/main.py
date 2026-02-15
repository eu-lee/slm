from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIR
from backend.database import init_db
from backend.inference.engine import ModelEngine
from backend.routers import auth, chat, conversations


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init database tables
    await init_db()
    # Load model at startup
    app.state.engine = ModelEngine()
    yield


app = FastAPI(title="SLM Chat", lifespan=lifespan)

# CORS for local dev (Next.js dev server on :3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(conversations.router)


# Health check
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve static frontend build (if it exists)
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
