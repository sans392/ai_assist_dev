"""In-memory log collector for the admin panel."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    timestamp: float
    level: str
    logger_name: str
    message: str
    is_ollama: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class LogCollector(logging.Handler):
    """Logging handler that stores recent log entries in memory."""

    def __init__(self, max_entries: int = 1000) -> None:
        super().__init__()
        self._entries: deque[LogEntry] = deque(maxlen=max_entries)

    def emit(self, record: logging.LogRecord) -> None:
        is_ollama = (
            record.name == "app.services.ollama"
            and ("Request payload" in record.getMessage()
                 or "Response:" in record.getMessage()
                 or "POST /api/chat" in record.getMessage())
        )
        entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            logger_name=record.name,
            message=self.format(record),
            is_ollama=is_ollama,
        )
        self._entries.append(entry)

    def get_entries(self, ollama_only: bool = False, limit: int = 200) -> list[dict]:
        entries = self._entries
        if ollama_only:
            entries = [e for e in entries if e.is_ollama]
        result = list(entries)[-limit:]
        return [e.to_dict() for e in result]


# Singleton instance
log_collector = LogCollector()
