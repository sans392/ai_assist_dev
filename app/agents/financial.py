from datetime import date

from app.agents.base import BaseAgent
from app.services.finance import FinanceService

_finance = FinanceService(today=date.today())


class FinancialAnalystAgent(BaseAgent):
    name = "financial"
    default_system_prompt = ""

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        # Update current date on every call (in case the server runs for days)
        _finance._today = date.today()

        summary = _finance.get_summary()

        # Extract last user message for context-aware analytics
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m["content"]
                break

        analytics = _finance.get_analytics(last_user_msg)

        parts = [self.system_prompt, "\n", summary]
        if analytics:
            parts.extend(["\n\n", analytics])
        system_content = "".join(parts)

        return [{"role": "system", "content": system_content}] + [
            m for m in messages if m.get("role") != "system"
        ]
