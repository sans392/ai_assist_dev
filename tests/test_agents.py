from app.agents import get_agent_by_mode
from app.agents.financial import FinancialAnalystAgent
from app.agents.general import GeneralAgent


def test_general_agent_prepends_system():
    agent = GeneralAgent()
    messages = [{"role": "user", "content": "Hello"}]
    result = agent.prepare_messages(messages)
    assert result[0]["role"] == "system"
    assert len(result) == 2


def test_general_agent_skips_if_system_present():
    agent = GeneralAgent()
    messages = [{"role": "system", "content": "Custom"}, {"role": "user", "content": "Hello"}]
    result = agent.prepare_messages(messages)
    assert result[0]["content"] == "Custom"
    assert len(result) == 2


def test_financial_agent_has_system_prompt():
    agent = FinancialAnalystAgent()
    messages = [{"role": "user", "content": "Show expenses"}]
    result = agent.prepare_messages(messages)
    assert "ФИНАНСОВАЯ СВОДКА" in result[0]["content"]


def test_get_agent_by_mode():
    assert isinstance(get_agent_by_mode("general"), GeneralAgent)
    assert isinstance(get_agent_by_mode("financial"), FinancialAnalystAgent)
