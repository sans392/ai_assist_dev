from app.agents.base import BaseAgent
from app.services.finance import FinanceService

_finance = FinanceService()


class FinancialAnalystAgent(BaseAgent):
    name = "financial"
    default_system_prompt = (
        "Ты — финансовый аналитик. Ты помогаешь пользователю анализировать "
        "его финансовые данные, отвечаешь на вопросы о расходах, доходах, "
        "сбережениях и инвестициях, даёшь практические рекомендации.\n\n"
        "Правила:\n"
        "- Будь точен в цифрах, всегда указывай валюту (RUB).\n"
        "- Ссылайся на конкретные транзакции, когда это уместно.\n"
        "- Когда пользователь спрашивает о тенденциях — сравнивай месяцы.\n"
        "- Отвечай на языке пользователя.\n"
        "- Всегда напоминай, что твои советы носят образовательный характер "
        "и не являются профессиональной финансовой консультацией.\n\n"
        "Ниже приведены финансовые данные пользователя:\n"
    )

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        summary = _finance.get_summary()
        system_content = self.system_prompt + "\n" + summary
        return [{"role": "system", "content": system_content}] + [
            m for m in messages if m.get("role") != "system"
        ]
