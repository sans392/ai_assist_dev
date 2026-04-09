"""Service for loading, filtering, and pre-computing financial analytics.

Key design principle: all math is done in Python — the LLM only presents
pre-computed results in natural language.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "transactions.json"

# ---------------------------------------------------------------------------
# Russian keyword stems → category names in data
# ---------------------------------------------------------------------------
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "salary": ["зарплат", "зп ", "оклад", "salary"],
    "rent": ["аренд", "квартир", "жильё", "жилье", "rent"],
    "groceries": ["продукт", "магазин", "супермаркет", "groceries", "еда", "еду"],
    "restaurant": [
        "ресторан", "кафе", "обед", "ужин", "доставк", "restaurant",
        "фастфуд", "fast food",
    ],
    "transport": [
        "транспорт", "такси", "метро", "автобус", "поезд",
        "каршеринг", "transport", "бензин", "топлив",
    ],
    "subscriptions": ["подписк", "subscription"],
    "clothing": ["одежд", "обувь", "clothing"],
    "entertainment": ["развлечен", "кино", "театр", "концерт", "entertainment"],
    "health": [
        "здоровь", "медицин", "врач", "аптек", "лекарств", "спортзал",
        "health", "стоматолог", "зубн", "дантист", "окулист", "терапевт",
        "витамин", "клиник", "больниц", "анализ",
    ],
    "education": ["образован", "курс", "обучен", "книг", "education"],
    "gifts": ["подарк", "цвет", "gifts"],
    "electronics": ["электроник", "техник", "гаджет", "наушник", "electronics"],
    "savings": ["сбережен", "накоплен", "накопительн", "savings"],
    "freelance": ["фриланс", "подработк", "freelance"],
}

# Extra keywords that trigger a description-level search (not just category)
_DESCRIPTION_KEYWORDS: list[str] = [
    "стоматолог", "дантист", "зубн",
    "окулист", "терапевт", "витамин",
    "кредит", "ипотек", "долг", "займ", "рассрочк",
    "тройк", "каршеринг",
    "спортзал", "фитнес",
    "netflix", "spotify", "youtube", "яндекс плюс",
]

# Russian month name stems → month number
_MONTH_STEMS: list[tuple[str, str]] = [
    ("январ", "01"), ("феврал", "02"), ("март", "03"), ("марте", "03"),
    ("апрел", "04"), ("мая", "05"), ("май", "05"), ("мае", "05"),
    ("июн", "06"), ("июл", "07"), ("август", "08"),
    ("сентябр", "09"), ("октябр", "10"), ("ноябр", "11"), ("декабр", "12"),
]

# How many individual transactions to inject into the prompt at most
_RELEVANT_TX_LIMIT = 30

# Russian month names for display
_MONTH_NAMES_RU = {
    "01": "январь", "02": "февраль", "03": "март", "04": "апрель",
    "05": "май", "06": "июнь", "07": "июль", "08": "август",
    "09": "сентябрь", "10": "октябрь", "11": "ноябрь", "12": "декабрь",
}

# Category → Russian display name
_CATEGORY_NAMES_RU: dict[str, str] = {
    "salary": "Зарплата",
    "rent": "Аренда",
    "groceries": "Продукты",
    "restaurant": "Рестораны и кафе",
    "transport": "Транспорт",
    "subscriptions": "Подписки",
    "clothing": "Одежда",
    "entertainment": "Развлечения",
    "health": "Здоровье",
    "education": "Образование",
    "gifts": "Подарки",
    "electronics": "Электроника",
    "savings": "Сбережения",
    "freelance": "Фриланс",
}


def _fmt(amount: float) -> str:
    """Format amount with thousands separator."""
    return f"{amount:,.0f} ₽".replace(",", " ")


def _month_label(ym: str) -> str:
    """'2025-04' → 'апрель 2025'."""
    parts = ym.split("-")
    name = _MONTH_NAMES_RU.get(parts[1], parts[1])
    return f"{name} {parts[0]}"


def _cat_ru(cat: str) -> str:
    return _CATEGORY_NAMES_RU.get(cat, cat)


class FinanceService:
    def __init__(self, today: date | None = None) -> None:
        self._transactions: list[dict] = []
        self._today: date = today or date.today()
        self._load()

    def _load(self) -> None:
        if DATA_FILE.exists():
            self._transactions = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    @property
    def transactions(self) -> list[dict]:
        return self._transactions

    # ==================================================================
    # PUBLIC: compact summary (always in prompt)
    # ==================================================================

    def get_summary(self) -> str:
        """Build a compact Russian-language summary with pre-computed aggregates."""
        if not self._transactions:
            return "Финансовые данные отсутствуют."

        txns = self._transactions
        income_total = sum(t["amount"] for t in txns if t["type"] == "income")
        expense_total = sum(t["amount"] for t in txns if t["type"] == "expense")

        # By category (expenses)
        by_cat: dict[str, float] = defaultdict(float)
        by_cat_count: dict[str, int] = Counter()
        for t in txns:
            if t["type"] == "expense":
                by_cat[t["category"]] += t["amount"]
                by_cat_count[t["category"]] += 1

        # By month
        by_month: dict[str, dict[str, float]] = {}
        for t in txns:
            m = t["date"][:7]
            if m not in by_month:
                by_month[m] = {"income": 0.0, "expense": 0.0}
            by_month[m][t["type"]] += t["amount"]

        lines = [
            "=== ФИНАНСОВАЯ СВОДКА ===",
            f"Сегодняшняя дата: {self._today.isoformat()}",
            f"Период данных: {txns[0]['date']} — {txns[-1]['date']}",
            f"Всего транзакций: {len(txns)}",
            "",
            f"Общий доход: {_fmt(income_total)}",
            f"Общие расходы: {_fmt(expense_total)}",
            f"Баланс (доход − расход): {_fmt(income_total - expense_total)}",
            "",
            "--- Помесячная сводка ---",
        ]

        for ym in sorted(by_month):
            m = by_month[ym]
            bal = m["income"] - m["expense"]
            sign = "+" if bal >= 0 else ""
            lines.append(
                f"  {_month_label(ym)}: доход {_fmt(m['income'])} | "
                f"расход {_fmt(m['expense'])} | баланс {sign}{_fmt(bal)}"
            )

        lines.append("")
        lines.append("--- Расходы по категориям (за весь период) ---")
        for cat, amount in sorted(by_cat.items(), key=lambda x: -x[1]):
            lines.append(
                f"  {_cat_ru(cat)}: {_fmt(amount)} ({by_cat_count[cat]} операций)"
            )

        return "\n".join(lines)

    # ==================================================================
    # PUBLIC: pre-computed analytics based on user question
    # ==================================================================

    def get_analytics(self, user_message: str) -> str:
        """Analyse user question and return pre-computed answer as structured text.

        The LLM should present these numbers in natural language — no need
        to re-calculate anything.
        """
        if not self._transactions:
            return ""

        text = user_message.lower()
        sections: list[str] = []

        # --- Detect intent signals ---
        categories = self._detect_categories(text)
        months = self._detect_months(text)
        tx_type = self._detect_type(text)
        desc_keywords = self._detect_description_keywords(text)
        date_range = self._detect_date_range(text)
        wants_time_analysis = self._wants_time_of_day(text)
        wants_full_analysis = self._wants_full_analysis(text)
        wants_savings_advice = self._wants_savings_advice(text)
        wants_balance = self._wants_balance(text)
        wants_loans = self._wants_loans(text)

        # --- Full analysis ---
        if wants_full_analysis:
            return self._build_full_analysis()

        # --- Loan search ---
        if wants_loans:
            sections.append(self._build_loan_search())

        # --- Time-of-day analysis ---
        if wants_time_analysis:
            sections.append(self._build_time_of_day(months, categories, tx_type))

        # --- Description-level search (e.g. "стоматолог") ---
        if desc_keywords:
            sections.append(self._build_description_search(desc_keywords, months, date_range))

        # --- Category + period analytics ---
        if categories or months or tx_type or date_range:
            sections.append(
                self._build_filtered_analytics(categories, months, tx_type, date_range)
            )

        # --- Balance query ---
        if wants_balance and not sections:
            sections.append(self._build_balance(months, date_range))

        # --- Savings advice ---
        if wants_savings_advice:
            sections.append(self._build_savings_advice())

        # --- Fallback: if nothing matched, provide recent summary ---
        if not sections:
            sections.append(self._build_recent_summary())

        return "\n\n".join(s for s in sections if s)

    # Keep backward-compatible method
    def get_relevant_context(self, user_message: str, limit: int = _RELEVANT_TX_LIMIT) -> str:
        """Backward-compatible wrapper — now delegates to get_analytics."""
        return self.get_analytics(user_message)

    # ==================================================================
    # PUBLIC: filter method (used by tests, admin, etc.)
    # ==================================================================

    def filter_transactions(
        self,
        categories: set[str] | None = None,
        months: set[str] | None = None,
        tx_type: str | None = None,
    ) -> list[dict]:
        return self._filter(categories, months, tx_type)

    # ==================================================================
    # ANALYTICS BUILDERS
    # ==================================================================

    def _build_filtered_analytics(
        self,
        categories: set[str] | None,
        months: set[str] | None,
        tx_type: str | None,
        date_range: tuple[date, date] | None,
    ) -> str:
        filtered = self._filter(categories, months, tx_type)
        if date_range:
            d_from, d_to = date_range
            filtered = [
                t for t in filtered
                if d_from <= date.fromisoformat(t["date"]) <= d_to
            ]

        if not filtered:
            parts = []
            if categories:
                parts.append(f"категории: {', '.join(_cat_ru(c) for c in categories)}")
            if months:
                parts.append(f"месяцы: {', '.join(_month_label(m) for m in sorted(months))}")
            if tx_type:
                parts.append(f"тип: {'доход' if tx_type == 'income' else 'расход'}")
            if date_range:
                parts.append(f"период: {date_range[0]} — {date_range[1]}")
            return f"--- Данные не найдены (фильтры: {'; '.join(parts)}) ---"

        income = sum(t["amount"] for t in filtered if t["type"] == "income")
        expense = sum(t["amount"] for t in filtered if t["type"] == "expense")
        total_count = len(filtered)

        # Build header
        filter_desc = []
        if categories:
            filter_desc.append(
                f"Категории: {', '.join(_cat_ru(c) for c in sorted(categories))}"
            )
        if months:
            filter_desc.append(
                f"Период: {', '.join(_month_label(m) for m in sorted(months))}"
            )
        if date_range:
            filter_desc.append(f"Период: {date_range[0]} — {date_range[1]}")
        if tx_type:
            filter_desc.append(f"Тип: {'доход' if tx_type == 'income' else 'расход'}")

        lines = [f"--- РЕЗУЛЬТАТ ЗАПРОСА ({'; '.join(filter_desc)}) ---"]
        lines.append(f"Найдено транзакций: {total_count}")

        if tx_type == "income" or (income > 0 and expense == 0):
            lines.append(f"Сумма доходов: {_fmt(income)}")
        elif tx_type == "expense" or (expense > 0 and income == 0):
            lines.append(f"Сумма расходов: {_fmt(expense)}")
        else:
            lines.append(f"Доходы: {_fmt(income)}")
            lines.append(f"Расходы: {_fmt(expense)}")
            lines.append(f"Баланс: {_fmt(income - expense)}")

        # Breakdown by category
        by_cat: dict[str, float] = defaultdict(float)
        by_cat_count: dict[str, int] = Counter()
        for t in filtered:
            by_cat[t["category"]] += t["amount"]
            by_cat_count[t["category"]] += 1

        if len(by_cat) > 1:
            lines.append("")
            lines.append("По категориям:")
            for cat, amount in sorted(by_cat.items(), key=lambda x: -x[1]):
                lines.append(
                    f"  {_cat_ru(cat)}: {_fmt(amount)} ({by_cat_count[cat]} оп.)"
                )

        # Top transactions (up to 10)
        top = sorted(filtered, key=lambda t: t["amount"], reverse=True)[:10]
        if top:
            lines.append("")
            lines.append(f"Крупнейшие операции (топ-{len(top)}):")
            for t in top:
                sign = "+" if t["type"] == "income" else "−"
                lines.append(
                    f"  {t['date']}  {sign}{_fmt(t['amount'])}  "
                    f"[{_cat_ru(t['category'])}] {t['description']}"
                )

        return "\n".join(lines)

    def _build_description_search(
        self,
        keywords: list[str],
        months: set[str] | None,
        date_range: tuple[date, date] | None,
    ) -> str:
        """Search transactions by description keywords."""
        matched: list[dict] = []
        for t in self._transactions:
            desc_lower = t["description"].lower()
            if any(kw in desc_lower for kw in keywords):
                matched.append(t)

        if months:
            matched = [t for t in matched if t["date"][:7] in months]
        if date_range:
            d_from, d_to = date_range
            matched = [
                t for t in matched
                if d_from <= date.fromisoformat(t["date"]) <= d_to
            ]

        kw_display = ", ".join(keywords)
        if not matched:
            return f"--- Поиск по описанию «{kw_display}»: ничего не найдено ---"

        total = sum(t["amount"] for t in matched)
        income = sum(t["amount"] for t in matched if t["type"] == "income")
        expense = sum(t["amount"] for t in matched if t["type"] == "expense")

        lines = [f"--- Поиск по описанию «{kw_display}» ---"]
        lines.append(f"Найдено: {len(matched)} транзакций на сумму {_fmt(total)}")
        if income > 0 and expense > 0:
            lines.append(f"  Доходы: {_fmt(income)}, Расходы: {_fmt(expense)}")

        lines.append("")
        lines.append("Все совпадения:")
        for t in matched:
            sign = "+" if t["type"] == "income" else "−"
            lines.append(
                f"  {t['date']}  {sign}{_fmt(t['amount'])}  "
                f"[{_cat_ru(t['category'])}] {t['description']}"
            )

        return "\n".join(lines)

    def _build_time_of_day(
        self,
        months: set[str] | None,
        categories: set[str] | None,
        tx_type: str | None,
    ) -> str:
        """Analyse purchase distribution by time of day."""
        filtered = self._filter(categories, months, tx_type or "expense")

        # Group by hour
        by_hour: dict[int, int] = Counter()
        by_hour_amount: dict[int, float] = defaultdict(float)
        for t in filtered:
            hour = int(t["time"].split(":")[0])
            by_hour[hour] += 1
            by_hour_amount[hour] += t["amount"]

        if not by_hour:
            return "--- Анализ по времени суток: нет данных ---"

        lines = ["--- АНАЛИЗ ПО ВРЕМЕНИ СУТОК ---"]
        lines.append(f"Всего операций в выборке: {sum(by_hour.values())}")
        lines.append("")
        lines.append("Топ-10 часов по количеству операций:")
        for hour, count in sorted(by_hour.items(), key=lambda x: -x[1])[:10]:
            avg = by_hour_amount[hour] / count
            lines.append(
                f"  {hour:02d}:00–{hour:02d}:59  —  {count} операций, "
                f"сумма {_fmt(by_hour_amount[hour])}, средний чек {_fmt(avg)}"
            )

        # Time ranges
        morning = sum(by_hour.get(h, 0) for h in range(6, 12))
        afternoon = sum(by_hour.get(h, 0) for h in range(12, 18))
        evening = sum(by_hour.get(h, 0) for h in range(18, 24))
        night = sum(by_hour.get(h, 0) for h in range(0, 6))
        total = morning + afternoon + evening + night

        if total > 0:
            lines.append("")
            lines.append("По времени суток:")
            for label, val in [
                ("Утро (06–12)", morning),
                ("День (12–18)", afternoon),
                ("Вечер (18–00)", evening),
                ("Ночь (00–06)", night),
            ]:
                pct = val / total * 100
                lines.append(f"  {label}: {val} операций ({pct:.0f}%)")

        return "\n".join(lines)

    def _build_loan_search(self) -> str:
        """Search for loan/credit related transactions."""
        loan_kw = ["кредит", "ипотек", "долг", "займ", "рассрочк", "взнос по"]
        matched: list[dict] = []
        for t in self._transactions:
            desc = t["description"].lower()
            if any(kw in desc for kw in loan_kw):
                matched.append(t)

        if not matched:
            return (
                "--- Поиск кредитов/долгов ---\n"
                "В данных не найдено транзакций, связанных с кредитами, "
                "ипотекой, займами или рассрочками."
            )

        total = sum(t["amount"] for t in matched)
        lines = ["--- Кредиты и долги ---"]
        lines.append(f"Найдено: {len(matched)} транзакций на сумму {_fmt(total)}")
        for t in matched:
            sign = "+" if t["type"] == "income" else "−"
            lines.append(
                f"  {t['date']}  {sign}{_fmt(t['amount'])}  {t['description']}"
            )
        return "\n".join(lines)

    def _build_balance(
        self,
        months: set[str] | None,
        date_range: tuple[date, date] | None,
    ) -> str:
        filtered = self._transactions
        if months:
            filtered = [t for t in filtered if t["date"][:7] in months]
        if date_range:
            d_from, d_to = date_range
            filtered = [
                t for t in filtered
                if d_from <= date.fromisoformat(t["date"]) <= d_to
            ]

        income = sum(t["amount"] for t in filtered if t["type"] == "income")
        expense = sum(t["amount"] for t in filtered if t["type"] == "expense")

        period = "за весь период"
        if months:
            period = f"за {', '.join(_month_label(m) for m in sorted(months))}"
        if date_range:
            period = f"за {date_range[0]} — {date_range[1]}"

        lines = [f"--- БАЛАНС ({period}) ---"]
        lines.append(f"Доходы: {_fmt(income)}")
        lines.append(f"Расходы: {_fmt(expense)}")
        lines.append(f"Баланс (доход − расход): {_fmt(income - expense)}")
        return "\n".join(lines)

    def _build_savings_advice(self) -> str:
        """Pre-compute data for savings recommendations."""
        txns = self._transactions
        expenses = [t for t in txns if t["type"] == "expense"]

        by_cat: dict[str, float] = defaultdict(float)
        for t in expenses:
            by_cat[t["category"]] += t["amount"]

        total_expense = sum(by_cat.values())
        months_count = len({t["date"][:7] for t in txns})
        avg_monthly = total_expense / max(months_count, 1)

        # Find categories with high share
        lines = ["--- ДАННЫЕ ДЛЯ РЕКОМЕНДАЦИЙ ПО ЭКОНОМИИ ---"]
        lines.append(f"Средние ежемесячные расходы: {_fmt(avg_monthly)}")
        lines.append("")
        lines.append("Расходы по категориям (доля от общих):")
        for cat, amount in sorted(by_cat.items(), key=lambda x: -x[1]):
            pct = amount / total_expense * 100
            monthly_avg = amount / max(months_count, 1)
            lines.append(
                f"  {_cat_ru(cat)}: {_fmt(amount)} ({pct:.1f}%), "
                f"~{_fmt(monthly_avg)}/мес"
            )

        # Month-over-month volatility for top categories
        by_cat_month: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for t in expenses:
            by_cat_month[t["category"]][t["date"][:7]] += t["amount"]

        lines.append("")
        lines.append("Помесячная динамика крупнейших категорий:")
        for cat in sorted(by_cat, key=lambda c: -by_cat[c])[:5]:
            month_vals = by_cat_month[cat]
            vals = [month_vals.get(m, 0) for m in sorted({t["date"][:7] for t in txns})]
            if vals:
                mn, mx = min(vals), max(vals)
                lines.append(
                    f"  {_cat_ru(cat)}: мин {_fmt(mn)}/мес, макс {_fmt(mx)}/мес"
                )

        return "\n".join(lines)

    def _build_full_analysis(self) -> str:
        """Comprehensive pre-computed analysis."""
        sections = [
            self.get_summary(),
            self._build_balance(None, None),
            self._build_time_of_day(None, None, "expense"),
            self._build_savings_advice(),
        ]

        # Top-10 largest expenses
        expenses = sorted(
            (t for t in self._transactions if t["type"] == "expense"),
            key=lambda t: t["amount"],
            reverse=True,
        )[:10]
        lines = ["--- ТОП-10 КРУПНЕЙШИХ РАСХОДОВ ---"]
        for t in expenses:
            lines.append(
                f"  {t['date']}  −{_fmt(t['amount'])}  "
                f"[{_cat_ru(t['category'])}] {t['description']}"
            )
        sections.append("\n".join(lines))

        # Top-10 income sources
        incomes = sorted(
            (t for t in self._transactions if t["type"] == "income"),
            key=lambda t: t["amount"],
            reverse=True,
        )[:10]
        lines = ["--- ТОП-10 КРУПНЕЙШИХ ДОХОДОВ ---"]
        for t in incomes:
            lines.append(
                f"  {t['date']}  +{_fmt(t['amount'])}  "
                f"[{_cat_ru(t['category'])}] {t['description']}"
            )
        sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _build_recent_summary(self) -> str:
        """Fallback: summary of last 30 days."""
        d_to = self._today
        d_from = d_to - timedelta(days=30)
        recent = [
            t for t in self._transactions
            if d_from <= date.fromisoformat(t["date"]) <= d_to
        ]
        if not recent:
            # Fallback to last 10 transactions
            recent = self._transactions[-10:]

        income = sum(t["amount"] for t in recent if t["type"] == "income")
        expense = sum(t["amount"] for t in recent if t["type"] == "expense")

        lines = [f"--- ПОСЛЕДНИЕ 30 ДНЕЙ ({d_from} — {d_to}) ---"]
        lines.append(f"Транзакций: {len(recent)}")
        lines.append(f"Доходы: {_fmt(income)}")
        lines.append(f"Расходы: {_fmt(expense)}")
        lines.append(f"Баланс: {_fmt(income - expense)}")

        # Last 10 transactions
        lines.append("")
        lines.append("Последние операции:")
        for t in recent[-10:]:
            sign = "+" if t["type"] == "income" else "−"
            lines.append(
                f"  {t['date']} {t['time']}  {sign}{_fmt(t['amount'])}  "
                f"[{_cat_ru(t['category'])}] {t['description']}"
            )

        return "\n".join(lines)

    # ==================================================================
    # INTENT DETECTORS
    # ==================================================================

    def _detect_categories(self, text: str) -> set[str]:
        found: set[str] = set()
        for category, keywords in _CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    found.add(category)
                    break
        return found

    def _detect_description_keywords(self, text: str) -> list[str]:
        """Detect keywords that require description-level search."""
        found = []
        for kw in _DESCRIPTION_KEYWORDS:
            if kw in text:
                found.append(kw)
        return found

    def _detect_months(self, text: str) -> set[str]:
        """Detect month references with relative time support."""
        found_numbers: set[str] = set()

        # Direct YYYY-MM pattern
        for m in re.findall(r"\d{4}-\d{2}", text):
            found_numbers.add(m)

        # Russian month name stems
        matched_month_nums: list[str] = []
        for stem, num in _MONTH_STEMS:
            if stem in text:
                matched_month_nums.append(num)

        # Determine year context
        is_last_year = "прошл" in text and ("год" in text or "лет" in text)
        current_year = str(self._today.year)
        previous_year = str(self._today.year - 1)

        # "прошлый месяц" / "прошлом месяце" — relative
        if ("прошл" in text) and ("месяц" in text) and not matched_month_nums:
            prev = self._today.replace(day=1) - timedelta(days=1)
            found_numbers.add(f"{prev.year}-{prev.month:02d}")

        # "этот месяц" / "текущий месяц"
        if any(kw in text for kw in ["этот месяц", "этом месяц", "текущ"]):
            found_numbers.add(f"{self._today.year}-{self._today.month:02d}")

        # Named months
        for num in matched_month_nums:
            if is_last_year:
                found_numbers.add(f"{previous_year}-{num}")
            else:
                # Default to the year where this month exists in data
                # Prefer current year, fall back to any year in data
                years_in_data = {t["date"][:4] for t in self._transactions}
                target_ym = f"{current_year}-{num}"
                prev_ym = f"{previous_year}-{num}"
                if any(t["date"][:7] == target_ym for t in self._transactions):
                    found_numbers.add(target_ym)
                elif any(t["date"][:7] == prev_ym for t in self._transactions):
                    found_numbers.add(prev_ym)
                else:
                    for year in sorted(years_in_data, reverse=True):
                        found_numbers.add(f"{year}-{num}")

        return found_numbers

    def _detect_date_range(self, text: str) -> tuple[date, date] | None:
        """Detect relative day ranges like 'последние 40 дней'."""
        m = re.search(r"последн\w*\s+(\d+)\s*дн", text)
        if m:
            days = int(m.group(1))
            return (self._today - timedelta(days=days), self._today)

        m = re.search(r"за\s+(\d+)\s*дн", text)
        if m:
            days = int(m.group(1))
            return (self._today - timedelta(days=days), self._today)

        return None

    def _detect_type(self, text: str) -> str | None:
        income_kw = ["доход", "заработ", "получил", "поступлен", "income", "зарплат"]
        expense_kw = [
            "расход", "потратил", "потратила", "трат", "покупк",
            "expense", "купил", "купила", "оплат",
        ]

        has_income = any(kw in text for kw in income_kw)
        has_expense = any(kw in text for kw in expense_kw)

        if has_income and not has_expense:
            return "income"
        if has_expense and not has_income:
            return "expense"
        return None

    def _wants_time_of_day(self, text: str) -> bool:
        return any(
            kw in text
            for kw in ["время суток", "время дня", "по часам", "в какое время",
                        "когда чаще", "по времени"]
        )

    def _wants_full_analysis(self, text: str) -> bool:
        return any(
            kw in text
            for kw in ["полный анализ", "полная аналитика", "полный отчёт",
                        "полный отчет", "общий анализ", "детальный анализ",
                        "подробный анализ"]
        )

    def _wants_savings_advice(self, text: str) -> bool:
        return any(
            kw in text
            for kw in ["сократить", "сэкономить", "экономи", "уменьшить расход",
                        "оптимизир", "как меньше тратить", "совет"]
        )

    def _wants_balance(self, text: str) -> bool:
        return any(kw in text for kw in ["баланс", "итог", "общий"])

    def _wants_loans(self, text: str) -> bool:
        return any(
            kw in text
            for kw in ["кредит", "ипотек", "долг", "займ", "рассрочк",
                        "погаш", "непогаш"]
        )

    # ==================================================================
    # INTERNAL: filter helper
    # ==================================================================

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
