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
    assert "Recent Transactions" in summary
    # Compact summary should NOT contain all transactions dump
    assert "All Transactions" not in summary


def test_finance_summary_compact_size():
    """Summary should stay small regardless of transaction count."""
    svc = FinanceService()
    summary = svc.get_summary()
    lines = summary.strip().split("\n")
    # Aggregates + 10 recent ≈ under 60 lines even with many months/categories
    assert len(lines) < 80


def test_financial_agent_injects_data():
    agent = FinancialAnalystAgent()
    messages = [{"role": "user", "content": "Сколько я потратил?"}]
    prepared = agent.prepare_messages(messages)

    system_msg = prepared[0]
    assert system_msg["role"] == "system"
    assert "Total expenses" in system_msg["content"]
    assert "RUB" in system_msg["content"]
    assert len(prepared) == 2  # system + user


# --- Filtering tests ---


def test_detect_categories_russian():
    svc = FinanceService()
    cats = svc._detect_categories("покажи расходы на рестораны")
    assert "restaurant" in cats


def test_detect_categories_multiple():
    svc = FinanceService()
    cats = svc._detect_categories("сколько на такси и продукты")
    assert "transport" in cats
    assert "groceries" in cats


def test_detect_months_russian():
    svc = FinanceService()
    months = svc._detect_months("расходы за январь")
    assert any(m.endswith("-01") for m in months)


def test_detect_months_iso_format():
    svc = FinanceService()
    months = svc._detect_months("данные за 2025-07")
    assert "2025-07" in months


def test_detect_type_expense():
    svc = FinanceService()
    assert svc._detect_type("сколько я потратил") == "expense"


def test_detect_type_income():
    svc = FinanceService()
    assert svc._detect_type("покажи мои доходы") == "income"


def test_detect_type_neutral():
    svc = FinanceService()
    assert svc._detect_type("привет") is None


def test_filter_by_category():
    svc = FinanceService()
    result = svc.filter_transactions(categories={"rent"})
    assert len(result) > 0
    assert all(t["category"] == "rent" for t in result)


def test_filter_by_month():
    svc = FinanceService()
    result = svc.filter_transactions(months={"2025-06"})
    assert len(result) > 0
    assert all(t["date"].startswith("2025-06") for t in result)


def test_filter_by_type():
    svc = FinanceService()
    result = svc.filter_transactions(tx_type="income")
    assert len(result) > 0
    assert all(t["type"] == "income" for t in result)


def test_filter_combined():
    svc = FinanceService()
    result = svc.filter_transactions(categories={"groceries"}, months={"2025-08"})
    assert all(t["category"] == "groceries" and t["date"].startswith("2025-08") for t in result)


def test_relevant_context_with_keywords():
    svc = FinanceService()
    ctx = svc.get_relevant_context("покажи расходы на рестораны в январе")
    assert "restaurant" in ctx
    assert "Relevant Transactions" in ctx


def test_relevant_context_respects_limit():
    svc = FinanceService()
    ctx = svc.get_relevant_context("все транзакции", limit=10)
    # Count individual transaction lines (start with "  20")
    tx_lines = [line for line in ctx.split("\n") if line.startswith("  20")]
    assert len(tx_lines) <= 10


def test_relevant_context_no_keywords_returns_top():
    """When no specific keywords match, return top transactions by amount."""
    svc = FinanceService()
    ctx = svc.get_relevant_context("привет")
    assert "Relevant Transactions" in ctx


def test_agent_injects_relevant_context():
    """Agent should include filtered transactions based on user message."""
    agent = FinancialAnalystAgent()
    messages = [{"role": "user", "content": "Сколько я потратил на такси?"}]
    prepared = agent.prepare_messages(messages)
    system_content = prepared[0]["content"]
    assert "transport" in system_content
    assert "Relevant Transactions" in system_content
