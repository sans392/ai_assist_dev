from app.agents.financial import FinancialAnalystAgent
from app.services.finance import FinanceService


def test_finance_service_loads_data():
    svc = FinanceService()
    assert len(svc.transactions) > 0
    assert svc.transactions[0]["type"] in ("income", "expense")


def test_finance_summary_contains_key_sections():
    svc = FinanceService()
    summary = svc.get_summary()
    assert "Total income" in summary
    assert "Total expenses" in summary
    assert "By Month" in summary
    assert "Expenses by Category" in summary
    assert "All Transactions" in summary


def test_financial_agent_injects_data():
    agent = FinancialAnalystAgent()
    messages = [{"role": "user", "content": "Сколько я потратил?"}]
    prepared = agent.prepare_messages(messages)

    system_msg = prepared[0]
    assert system_msg["role"] == "system"
    assert "Total expenses" in system_msg["content"]
    assert "RUB" in system_msg["content"]
    assert len(prepared) == 2  # system + user
