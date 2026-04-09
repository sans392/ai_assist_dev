# AI Assistant Backend

Backend for an AI assistant learning project. Communicates with a local [Ollama](https://ollama.com/) instance.

## Quick Start (Docker)

Assumes Ollama is already running via Docker Compose on network `ollama-net`.

```bash
# Copy env config and set your Ollama container name
cp .env.example .env

# Start
docker compose up --build
```

API will be available at `http://localhost:8000`. Code changes in `app/` apply instantly (hot-reload via volume mount).

## Quick Start (Local)

```bash
pip install -e ".[dev]"
cp .env.example .env
# Set OLLAMA_BASE_URL=http://localhost:11434 in .env
uvicorn app.main:app --reload
```

## API

### `POST /chat`
Send a chat message. Supports `mode` selection (`general`, `financial`) and streaming via SSE.

```json
{
  "message": "Hello!",
  "mode": "general",
  "stream": false,
  "conversation_id": null
}
```

### `GET /health`
Check API and Ollama connectivity status.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
app/
├── main.py           # FastAPI app, lifespan, CORS
├── config.py         # Settings from environment
├── dependencies.py   # FastAPI dependency injection
├── routers/          # API endpoints (chat, health, admin)
├── services/         # Ollama client, conversation store, finance analytics
├── agents/           # Agent modes (general, financial)
└── models/           # Pydantic schemas
data/
├── transactions.json     # Financial transaction data (1000+ records)
└── agent_defaults.json   # Per-agent settings (model, prompt, options)
```
