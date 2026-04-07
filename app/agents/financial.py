from app.agents.base import BaseAgent


class FinancialAnalystAgent(BaseAgent):
    name = "financial"
    system_prompt = (
        "You are a financial analyst AI assistant. "
        "You help users analyze their financial data, answer questions about spending, "
        "income, savings, and investments, and provide actionable recommendations. "
        "Be precise with numbers. When you lack data, ask the user for clarification. "
        "Always remind the user that your advice is educational, not professional financial advice."
    )

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        # TODO: inject user's financial data context here when data layer is ready
        return self._prepend_system(messages)
