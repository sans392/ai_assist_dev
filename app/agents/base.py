from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all assistant agents/modes."""

    name: str
    system_prompt: str

    @abstractmethod
    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Prepare message list before sending to the LLM.

        Typically prepends a system message and may inject additional context.
        """
        ...

    def _prepend_system(self, messages: list[dict]) -> list[dict]:
        """Helper: prepend system prompt if not already present."""
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages
