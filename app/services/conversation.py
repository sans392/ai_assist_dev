"""In-memory conversation storage. Swap for Redis/SQLite later."""


class ConversationStore:
    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}

    def get(self, conversation_id: str) -> list[dict]:
        return self._store.get(conversation_id, [])

    def append(self, conversation_id: str, message: dict) -> None:
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store[conversation_id].append(message)

    def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)
