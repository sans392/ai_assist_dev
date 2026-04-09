from datetime import date

from app.agents.financial import FinancialAnalystAgent
from app.services.finance import FinanceService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(today: str = "2026-04-09") -> FinanceService:
    """Create a FinanceService with a fixed 'today' for deterministic tests."""
    return FinanceService(today=date.fromisoformat(today))


# ---------------------------------------------------------------------------
# Basic loading & summary
# ---------------------------------------------------------------------------


def test_finance_service_loads_data():
    svc = _svc()
    assert len(svc.transactions) > 0
    assert svc.transactions[0]["type"] in ("income", "expense")


def test_finance_summary_contains_key_sections():
    svc = _svc()
    summary = svc.get_summary()
    assert "ФИНАНСОВАЯ СВОДКА" in summary
    assert "Общий доход" in summary
    assert "Общие расходы" in summary
    assert "Помесячная сводка" in summary
    assert "Расходы по категориям" in summary
    assert "₽" in summary


def test_finance_summary_compact_size():
    """Summary should stay small regardless of transaction count."""
    svc = _svc()
    summary = svc.get_summary()
    lines = summary.strip().split("\n")
    assert len(lines) < 80


def test_finance_summary_includes_today():
    svc = _svc("2026-04-09")
    summary = svc.get_summary()
    assert "2026-04-09" in summary


def test_financial_agent_injects_data():
    agent = FinancialAnalystAgent()
    messages = [{"role": "user", "content": "Сколько я потратил?"}]
    prepared = agent.prepare_messages(messages)

    system_msg = prepared[0]
    assert system_msg["role"] == "system"
    assert "₽" in system_msg["content"]
    assert len(prepared) == 2  # system + user


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------


def test_detect_categories_russian():
    svc = _svc()
    cats = svc._detect_categories("покажи расходы на рестораны")
    assert "restaurant" in cats


def test_detect_categories_multiple():
    svc = _svc()
    cats = svc._detect_categories("сколько на такси и продукты")
    assert "transport" in cats
    assert "groceries" in cats


def test_detect_categories_dentist():
    """'стоматолог' should trigger health category."""
    svc = _svc()
    cats = svc._detect_categories("сколько я потратил на стоматолога")
    assert "health" in cats


# ---------------------------------------------------------------------------
# Month detection (with relative dates)
# ---------------------------------------------------------------------------


def test_detect_months_russian():
    svc = _svc()
    months = svc._detect_months("расходы за январь")
    assert any(m.endswith("-01") for m in months)


def test_detect_months_iso_format():
    svc = _svc()
    months = svc._detect_months("данные за 2025-07")
    assert "2025-07" in months


def test_detect_months_last_month():
    """'прошлый месяц' relative to today=2026-04-09 → 2026-03."""
    svc = _svc("2026-04-09")
    months = svc._detect_months("сколько я потратил в прошлом месяце")
    assert "2026-03" in months


def test_detect_months_last_year():
    """'февраль в прошлом году' relative to today=2026 → 2025-02."""
    svc = _svc("2026-04-09")
    months = svc._detect_months("сколько я потратил в феврале в прошлом году")
    assert "2025-02" in months
    assert "2026-02" not in months


def test_detect_months_february_default_current_year():
    """'февраль' without year context → prefer current year (2026)."""
    svc = _svc("2026-04-09")
    months = svc._detect_months("расходы за февраль")
    assert "2026-02" in months


# ---------------------------------------------------------------------------
# Date range detection
# ---------------------------------------------------------------------------


def test_detect_date_range_last_n_days():
    svc = _svc("2026-04-09")
    r = svc._detect_date_range("мои доходы за последние 40 дней")
    assert r is not None
    assert r[0] == date(2026, 2, 28)
    assert r[1] == date(2026, 4, 9)


def test_detect_date_range_za_n_days():
    svc = _svc("2026-04-09")
    r = svc._detect_date_range("расходы за 10 дней")
    assert r is not None
    assert r[0] == date(2026, 3, 30)


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def test_detect_type_expense():
    svc = _svc()
    assert svc._detect_type("сколько я потратил") == "expense"


def test_detect_type_income():
    svc = _svc()
    assert svc._detect_type("покажи мои доходы") == "income"


def test_detect_type_neutral():
    svc = _svc()
    assert svc._detect_type("привет") is None


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------


def test_wants_time_of_day():
    svc = _svc()
    assert svc._wants_time_of_day("в какое время суток я чаще совершаю покупки")
    assert not svc._wants_time_of_day("сколько я потратил")


def test_wants_full_analysis():
    svc = _svc()
    assert svc._wants_full_analysis("полный анализ")
    assert not svc._wants_full_analysis("расходы за месяц")


def test_wants_loans():
    svc = _svc()
    assert svc._wants_loans("у меня остались не погашенные кредиты")
    assert not svc._wants_loans("расходы на еду")


def test_wants_savings_advice():
    svc = _svc()
    assert svc._wants_savings_advice("как сократить расходы")
    assert not svc._wants_savings_advice("расходы за февраль")


def test_wants_balance():
    svc = _svc()
    assert svc._wants_balance("общий баланс доходы + расходы за все время")


# ---------------------------------------------------------------------------
# Description keyword detection
# ---------------------------------------------------------------------------


def test_detect_description_keywords():
    svc = _svc()
    kw = svc._detect_description_keywords("сколько я потратил на стоматолога")
    assert "стоматолог" in kw


def test_detect_description_keywords_credit():
    svc = _svc()
    kw = svc._detect_description_keywords("есть ли у меня кредиты")
    assert "кредит" in kw


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_filter_by_category():
    svc = _svc()
    result = svc.filter_transactions(categories={"rent"})
    assert len(result) > 0
    assert all(t["category"] == "rent" for t in result)


def test_filter_by_month():
    svc = _svc()
    result = svc.filter_transactions(months={"2025-06"})
    assert len(result) > 0
    assert all(t["date"].startswith("2025-06") for t in result)


def test_filter_by_type():
    svc = _svc()
    result = svc.filter_transactions(tx_type="income")
    assert len(result) > 0
    assert all(t["type"] == "income" for t in result)


def test_filter_combined():
    svc = _svc()
    result = svc.filter_transactions(categories={"groceries"}, months={"2025-08"})
    assert all(
        t["category"] == "groceries" and t["date"].startswith("2025-08")
        for t in result
    )


# ---------------------------------------------------------------------------
# Analytics (pre-computed results)
# ---------------------------------------------------------------------------


def test_analytics_expense_query():
    """'потратил в феврале' → pre-computed expense total."""
    svc = _svc("2026-04-09")
    result = svc.get_analytics("сколько я потратил в феврале")
    assert "РЕЗУЛЬТАТ ЗАПРОСА" in result
    assert "₽" in result
    assert "расход" in result.lower()


def test_analytics_dentist():
    """Description search for 'стоматолог'."""
    svc = _svc()
    result = svc.get_analytics("сколько я потратил на стоматолога")
    assert "стоматолог" in result.lower()
    assert "₽" in result


def test_analytics_dentist_with_month():
    """Description search for dentist in a specific month."""
    svc = _svc("2026-04-09")
    result = svc.get_analytics("сколько я потратил на стоматолога в прошлом месяце")
    assert "стоматолог" in result.lower()


def test_analytics_loans():
    """Loan search."""
    svc = _svc()
    result = svc.get_analytics("у меня остались не погашенные кредиты")
    assert "кредит" in result.lower()


def test_analytics_income_last_40_days():
    """Income over last 40 days."""
    svc = _svc("2026-04-09")
    result = svc.get_analytics("мои доходы за последние 40 дней")
    assert "₽" in result


def test_analytics_balance():
    svc = _svc()
    result = svc.get_analytics("общий баланс доходы + расходы за все время")
    assert "БАЛАНС" in result
    assert "₽" in result


def test_analytics_time_of_day():
    svc = _svc()
    result = svc.get_analytics(
        "в какое время суток я чаще совершаю покупки, покажи топ-10"
    )
    assert "АНАЛИЗ ПО ВРЕМЕНИ СУТОК" in result
    assert "операций" in result


def test_analytics_savings_advice():
    svc = _svc()
    result = svc.get_analytics("как сократить расходы")
    assert "ЭКОНОМИИ" in result or "категориям" in result.lower()


def test_analytics_full_analysis():
    svc = _svc()
    result = svc.get_analytics("полный анализ")
    assert "ФИНАНСОВАЯ СВОДКА" in result
    assert "АНАЛИЗ ПО ВРЕМЕНИ СУТОК" in result
    assert "ЭКОНОМИИ" in result


def test_analytics_fallback():
    """Generic message → recent summary."""
    svc = _svc("2026-04-09")
    result = svc.get_analytics("привет")
    assert "ПОСЛЕДНИЕ 30 ДНЕЙ" in result


def test_agent_injects_analytics():
    """Agent should include pre-computed analytics based on user message."""
    agent = FinancialAnalystAgent()
    messages = [{"role": "user", "content": "Сколько я потратил на такси?"}]
    prepared = agent.prepare_messages(messages)
    system_content = prepared[0]["content"]
    assert "Транспорт" in system_content
    assert "₽" in system_content


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_get_relevant_context_delegates_to_analytics():
    """get_relevant_context should delegate to get_analytics."""
    svc = _svc()
    ctx = svc.get_relevant_context("покажи расходы на рестораны в январе")
    assert "₽" in ctx
