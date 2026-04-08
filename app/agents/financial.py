from app.agents.base import BaseAgent
from app.services.finance import FinanceService

_finance = FinanceService()


class FinancialAnalystAgent(BaseAgent):
    name = "financial"
    default_system_prompt = ""

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        summary = _finance.get_summary()

        # Extract last user message for context-aware filtering
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m["content"]
                break

        relevant = _finance.get_relevant_context(last_user_msg)
        parts = [self.system_prompt, "\n", summary]
        if relevant:
            parts.extend(["\n\n", relevant])
        system_content = "".join(parts)

        return [{"role": "system", "content": system_content}] + [
            m for m in messages if m.get("role") != "system"
        ]
