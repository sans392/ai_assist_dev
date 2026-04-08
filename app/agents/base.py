from abc import ABC, abstractmethod

from app.services.agent_settings import agent_settings_store


class BaseAgent(ABC):
    """Base class for all assistant agents/modes."""

    name: str
    default_system_prompt: str

    @property
    def model(self) -> str | None:
        """Return model override for this agent, or None for default."""
        settings = agent_settings_store.get(self.name)
        return settings.get("model")

    @property
    def system_prompt(self) -> str:
        """Return system prompt: custom from settings or default."""
        settings = agent_settings_store.get(self.name)
        return settings.get("system_prompt", self.default_system_prompt)

    @property
    def model_options(self) -> dict:
        """Return model options (temperature, top_p, num_ctx, etc.)."""
        settings = agent_settings_store.get(self.name)
        return settings.get("options", {})

    @abstractmethod
    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Prepare message list before sending to the LLM."""
        ...

    def _prepend_system(self, messages: list[dict]) -> list[dict]:
        """Helper: prepend system prompt if not already present."""
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages
