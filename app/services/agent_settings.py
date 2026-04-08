"""Runtime settings store — editable via admin panel."""

import json
from copy import deepcopy
from pathlib import Path

DEFAULTS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "agent_defaults.json"


def _load_defaults() -> dict:
    if DEFAULTS_FILE.exists():
        return json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
    return {}


class AgentSettingsStore:
    """In-memory store for per-agent settings, preloaded from agent_defaults.json."""

    def __init__(self) -> None:
        self._settings: dict[str, dict] = _load_defaults()

    def get(self, agent_name: str) -> dict:
        return deepcopy(self._settings.get(agent_name, {}))

    def get_all(self) -> dict:
        return deepcopy(self._settings)

    def update(self, agent_name: str, patch: dict) -> dict:
        if agent_name not in self._settings:
            self._settings[agent_name] = {}
        settings = self._settings[agent_name]

        if "system_prompt" in patch:
            settings["system_prompt"] = patch["system_prompt"]
        if "model" in patch:
            settings["model"] = patch["model"]
        if "options" in patch and isinstance(patch["options"], dict):
            settings.setdefault("options", {})
            settings["options"].update(patch["options"])

        return self.get(agent_name)


agent_settings_store = AgentSettingsStore()
