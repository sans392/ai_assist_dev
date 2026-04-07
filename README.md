# AI Assistant Backend

Backend for an AI assistant learning project. Communicates with a local [Ollama](https://ollama.com/) instance.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy env config
cp .env.example .env

# Make sure Ollama is running locally
ollama serve

# Start the server
uvicorn app.main:app --reload
```

## API

### `POST /chat`
Send a chat message. Supports `mode` selection (`general`, `financial`) and streaming via SSE.

```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "mode": "general",
  "stream": false
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
├── routers/          # API endpoints (chat, health)
├── services/         # Ollama client, conversation store
├── agents/           # Agent modes (general, financial)
└── models/           # Pydantic schemas
```
