from app.agents.base import BaseAgent


class GeneralAgent(BaseAgent):
    name = "general"
    default_system_prompt = (
        "Ты — полезный ИИ-ассистент. Отвечай чётко и по существу. "
        "Если чего-то не знаешь — честно скажи об этом. Отвечай на языке пользователя."
    )

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        return self._prepend_system(messages)
