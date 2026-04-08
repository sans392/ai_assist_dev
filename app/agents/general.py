from app.agents.base import BaseAgent


class GeneralAgent(BaseAgent):
    name = "general"
    default_system_prompt = ""

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        return self._prepend_system(messages)
