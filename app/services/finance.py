"""Service for loading and summarizing financial data."""

import json
import re
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "transactions.json"

# Russian keyword stems → category names in data
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "salary": ["зарплат", "зп ", "оклад", "salary"],
    "rent": ["аренд", "квартир", "жильё", "жилье", "rent"],
    "groceries": ["продукт", "магазин", "супермаркет", "groceries"],
    "restaurant": ["ресторан", "кафе", "обед", "ужин", "доставк", "restaurant"],
    "transport": ["транспорт", "такси", "метро", "автобус", "поезд", "каршеринг", "transport"],
    "subscriptions": ["подписк", "subscription"],
    "clothing": ["одежд", "обувь", "clothing"],
    "entertainment": ["развлечен", "кино", "театр", "концерт", "entertainment"],
    "health": ["здоровь", "медицин", "врач", "аптек", "лекарств", "спортзал", "health"],
    "education": ["образован", "курс", "обучен", "книг", "education"],
    "gifts": ["подарк", "цвет", "gifts"],
    "electronics": ["электроник", "техник", "гаджет", "наушник", "electronics"],
    "savings": ["сбережен", "накоплен", "накопительн", "savings"],
    "freelance": ["фриланс", "подработк", "freelance"],
}

# Russian month name stems → month number
_MONTH_STEMS: list[tuple[str, str]] = [
    ("январ", "01"), ("феврал", "02"), ("март", "03"),
    ("апрел", "04"), ("мая", "05"), ("май", "05"),
    ("июн", "06"), ("июл", "07"), ("август", "08"),
    ("сентябр", "09"), ("октябр", "10"), ("ноябр", "11"), ("декабр", "12"),
]

# How many individual transactions to inject into the prompt at most
_RELEVANT_TX_LIMIT = 50


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

    # ------------------------------------------------------------------
    # Compact summary (always included in the prompt)
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Build a compact text summary: aggregates only, no individual transactions."""
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
            month = t["date"][:7]
            if month not in by_month:
                by_month[month] = {"income": 0, "expense": 0}
            by_month[month][t["type"]] += t["amount"]

        lines = [
            "=== Financial Data Summary ===",
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

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Keyword-based filtering for relevant context
    # ------------------------------------------------------------------

    def get_relevant_context(self, user_message: str, limit: int = _RELEVANT_TX_LIMIT) -> str:
        """Return formatted transactions relevant to the user's question.

        Analyses the message for category, month, and type keywords,
        filters the dataset, and returns at most ``limit`` transactions.
        If no specific filters match, returns the largest transactions.
        """
        if not self._transactions:
            return ""

        text = user_message.lower()
        categories = self._detect_categories(text)
        months = self._detect_months(text)
        tx_type = self._detect_type(text)

        filtered = self._filter(categories, months, tx_type)

        has_filters = bool(categories or months or tx_type)

        if not has_filters:
            # No specific query → return top by amount (most notable)
            filtered = sorted(self._transactions, key=lambda t: t["amount"], reverse=True)

        if len(filtered) > limit:
            filtered = filtered[:limit]

        if not filtered:
            return ""

        header = f"--- Relevant Transactions ({len(filtered)} shown"
        if has_filters:
            parts = []
            if categories:
                parts.append(f"categories: {', '.join(sorted(categories))}")
            if months:
                parts.append(f"months: {', '.join(sorted(months))}")
            if tx_type:
                parts.append(f"type: {tx_type}")
            header += f"; filters: {'; '.join(parts)}"
        header += ") ---"

        lines = [header]
        for t in filtered:
            sign = "+" if t["type"] == "income" else "-"
            lines.append(
                f"  {t['date']} {t['time']}  {sign}{t['amount']:,.0f} RUB  "
                f"[{t['category']}] {t['description']}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public filter method
    # ------------------------------------------------------------------

    def filter_transactions(
        self,
        categories: set[str] | None = None,
        months: set[str] | None = None,
        tx_type: str | None = None,
    ) -> list[dict]:
        """Filter transactions by category, month (YYYY-MM), and/or type."""
        return self._filter(categories, months, tx_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(
        self,
        categories: set[str] | None,
        months: set[str] | None,
        tx_type: str | None,
    ) -> list[dict]:
        result = self._transactions
        if categories:
            result = [t for t in result if t["category"] in categories]
        if months:
            result = [t for t in result if t["date"][:7] in months]
        if tx_type:
            result = [t for t in result if t["type"] == tx_type]
        return result

    def _detect_categories(self, text: str) -> set[str]:
        found: set[str] = set()
        for category, keywords in _CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    found.add(category)
                    break
        return found

    def _detect_months(self, text: str) -> set[str]:
        """Detect month references and return as YYYY-MM strings."""
        found_numbers: set[str] = set()

        # Direct YYYY-MM pattern
        for m in re.findall(r"\d{4}-\d{2}", text):
            found_numbers.add(m)

        # Russian month name stems
        for stem, num in _MONTH_STEMS:
            if stem in text:
                found_numbers.add(num)

        if not found_numbers:
            return set()

        # Determine which years are present in data
        years_in_data = {t["date"][:4] for t in self._transactions}

        result: set[str] = set()
        for item in found_numbers:
            if len(item) == 7 and "-" in item:
                # Already YYYY-MM
                result.add(item)
            else:
                # Just month number — expand to all years in data
                for year in years_in_data:
                    result.add(f"{year}-{item}")

        return result

    def _detect_type(self, text: str) -> str | None:
        income_kw = ["доход", "заработ", "получил", "поступлен", "income"]
        expense_kw = ["расход", "потратил", "трат", "покупк", "expense"]

        has_income = any(kw in text for kw in income_kw)
        has_expense = any(kw in text for kw in expense_kw)

        # Only filter by type if one is mentioned but not both
        if has_income and not has_expense:
            return "income"
        if has_expense and not has_income:
            return "expense"
        return None
