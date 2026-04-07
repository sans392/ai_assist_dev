from app.agents.base import BaseAgent
from app.services.finance import FinanceService

_finance = FinanceService()


class FinancialAnalystAgent(BaseAgent):
    name = "financial"
    system_prompt = (
        "You are a financial analyst AI assistant. "
        "You help users analyze their financial data, answer questions about spending, "
        "income, savings, and investments, and provide actionable recommendations.\n\n"
        "Rules:\n"
        "- Be precise with numbers, always include currency (RUB).\n"
        "- Reference specific transactions when relevant.\n"
        "- When the user asks about trends, compare months.\n"
        "- Answer in the same language the user writes in.\n"
        "- Always remind that your advice is educational, not professional financial advice.\n\n"
        "Below is the user's financial data:\n\n"
    )

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        summary = _finance.get_summary()
        system_content = self.system_prompt + summary
        return [{"role": "system", "content": system_content}] + [
            m for m in messages if m.get("role") != "system"
        ]
