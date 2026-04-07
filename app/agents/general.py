from app.agents.base import BaseAgent


class GeneralAgent(BaseAgent):
    name = "general"
    system_prompt = (
        "You are a helpful AI assistant. Answer questions clearly and concisely. "
        "If you don't know something, say so honestly."
    )

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        return self._prepend_system(messages)
