"""Service for loading and summarizing financial data."""

import json
from pathlib import Path


DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "transactions.json"


class FinanceService:
    def __init__(self) -> None:
        self._transactions: list[dict] = []
        self._load()

    def _load(self) -> None:
        if DATA_FILE.exists():
            self._transactions = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    @property
    def transactions(self) -> list[dict]:
        return self._transactions

    def get_summary(self) -> str:
        """Build a text summary of financial data to inject into LLM context."""
        if not self._transactions:
            return "No financial data available."

        income = [t for t in self._transactions if t["type"] == "income"]
        expenses = [t for t in self._transactions if t["type"] == "expense"]

        total_income = sum(t["amount"] for t in income)
        total_expenses = sum(t["amount"] for t in expenses)

        # Group expenses by category
        by_category: dict[str, float] = {}
        for t in expenses:
            by_category[t["category"]] = by_category.get(t["category"], 0) + t["amount"]

        # Group by month
        by_month: dict[str, dict[str, float]] = {}
        for t in self._transactions:
            month = t["date"][:7]  # "2026-01"
            if month not in by_month:
                by_month[month] = {"income": 0, "expense": 0}
            by_month[month][t["type"]] += t["amount"]

        lines = [
            f"=== Financial Data Summary ===",
            f"Period: {self._transactions[0]['date']} — {self._transactions[-1]['date']}",
            f"Total transactions: {len(self._transactions)}",
            f"Total income: {total_income:,.0f} RUB",
            f"Total expenses: {total_expenses:,.0f} RUB",
            f"Balance: {total_income - total_expenses:,.0f} RUB",
            "",
            "--- By Month ---",
        ]

        for month in sorted(by_month):
            m = by_month[month]
            lines.append(
                f"  {month}: income {m['income']:,.0f} | expenses {m['expense']:,.0f} | "
                f"balance {m['income'] - m['expense']:,.0f}"
            )

        lines.append("")
        lines.append("--- Expenses by Category ---")
        for cat, amount in sorted(by_category.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {amount:,.0f} RUB")

        lines.append("")
        lines.append("--- Recent Transactions (last 10) ---")
        for t in self._transactions[-10:]:
            sign = "+" if t["type"] == "income" else "-"
            lines.append(
                f"  {t['date']} {t['time']}  {sign}{t['amount']:,.0f} RUB  "
                f"[{t['category']}] {t['description']}"
            )

        lines.append("")
        lines.append("--- All Transactions ---")
        for t in self._transactions:
            sign = "+" if t["type"] == "income" else "-"
            lines.append(
                f"  {t['date']} {t['time']}  {sign}{t['amount']:,.0f} RUB  "
                f"[{t['category']}] {t['description']}"
            )

        return "\n".join(lines)
