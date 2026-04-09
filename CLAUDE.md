# CLAUDE.md

## Project Overview

AI Assistant backend — learning project for building a Whoop-style AI assistant.
Backend only (FastAPI + Ollama). Frontend is handled by a separate team.

## Tech Stack

- **Python 3.11**, **FastAPI**, **httpx** (async Ollama client), **Pydantic v2**
- **Ollama** — local LLM, runs in Docker on network `ollama-net`
- **Docker Compose** — app connects to existing Ollama via external network
- No ORM, no database (yet), no auth, no LangChain

## Architecture

```
app/
├── main.py           # FastAPI app factory, lifespan, CORS, static files
├── config.py         # Pydantic BaseSettings from .env
├── dependencies.py   # FastAPI Depends() providers
├── routers/          # chat.py, health.py, admin.py
├── services/         # ollama.py, conversation.py, finance.py, agent_settings.py, log_collector.py
├── agents/           # base.py (ABC), general.py, financial.py
└── models/           # schemas.py (Pydantic request/response)
```

## Key Design Decisions

- **Agents pattern**: BaseAgent ABC with `prepare_messages()` + `model_options` + `model`. New agent = new file + register in `agents/__init__.py`. No frameworks.
- **Server-side conversation history**: Frontend sends only `message` + `conversation_id`. Server stores full history in ConversationStore (in-memory dict). History resets on mode switch.
- **Per-agent settings**: system_prompt, model, options (temperature, top_p, num_ctx) — stored in AgentSettingsStore, editable via admin panel at runtime, defaults from `data/agent_defaults.json`.
- **Streaming**: SSE (Server-Sent Events) via `StreamingResponse`, not WebSocket.
- **Financial agent**: pre-computes analytics in Python (sums, breakdowns, time-of-day stats) and injects results into system prompt. LLM presents data, never calculates. Supports relative dates ("прошлый месяц"), description search ("стоматолог"), loan detection, savings advice. Data from `data/transactions.json`.
- **Prompts in Russian** for better quality with small models (8b).

## Commands

```bash
# Run locally
uvicorn app.main:app --reload

# Run in Docker
docker compose up --build

# Run tests (no Ollama needed — mocked)
pytest tests/ -v

# Lint
ruff check app/ tests/
```

## URLs

- Chat UI: `http://localhost:8000`
- Admin panel: `http://localhost:8000/admin`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `GET /health`

## Code Style

- Keep it simple — no over-engineering, no speculative abstractions
- Type hints everywhere
- Async by default (httpx, FastAPI)
- Tests use `httpx.AsyncClient` + `ASGITransport`, Ollama calls are mocked via `unittest.mock`
- Commit messages: conventional commits in English (feat/fix/refactor)

## Environment

- `.env` for config (see `.env.example`)
- `OLLAMA_BASE_URL` — in Docker use container name (e.g. `http://ollama:11434`)
- `LOG_LEVEL=DEBUG` to see full Ollama request payloads
