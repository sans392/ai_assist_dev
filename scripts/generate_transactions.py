#!/usr/bin/env python3
"""Generate 1000+ realistic financial transactions for testing."""

import json
import random
from datetime import date, timedelta
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "transactions.json"

random.seed(42)

# --- Description templates per category (Russian) ---
DESCRIPTIONS: dict[str, list[str]] = {
    "salary": ["Зарплата за {prev_month}"],
    "freelance": [
        "Фриланс — вёрстка сайта",
        "Фриланс — доработка API",
        "Фриланс — дизайн баннера",
        "Фриланс — консультация",
        "Фриланс — тестирование",
    ],
    "rent": ["Аренда квартиры за {month_name}"],
    "groceries": [
        "Пятёрочка", "Перекрёсток", "Магнит", "ВкусВилл",
        "Лента", "Ашан", "Дикси", "Азбука Вкуса",
        "Продукты на неделю", "Фрукты и овощи",
    ],
    "restaurant": [
        "Обед в кафе", "Ужин в ресторане", "Бизнес-ланч",
        "Кофейня", "Доставка еды", "Суши-бар",
        "Пиццерия", "Фастфуд", "Бургерная",
    ],
    "transport": [
        "Метро", "Автобус", "Такси Яндекс", "Такси Uber",
        "Электричка", "Каршеринг", "Самокат", "Троллейбус",
        "Маршрутка", "Пополнение Тройки",
    ],
    "subscriptions": [
        "Подписка Яндекс Плюс", "Подписка Spotify", "Облачное хранилище",
        "VPN-сервис", "Подписка YouTube Premium", "Подписка Netflix",
    ],
    "clothing": [
        "Куртка зимняя", "Джинсы", "Кроссовки", "Футболка",
        "Рубашка", "Шапка и перчатки", "Свитер", "Обувь",
    ],
    "entertainment": [
        "Кинотеатр", "Концерт", "Боулинг", "Настольные игры",
        "Квест-комната", "Музей", "Выставка", "Театр",
        "Стриминг-донат", "Игра в Steam",
    ],
    "health": [
        "Аптека", "Приём терапевта", "Стоматолог", "Анализы",
        "Абонемент в спортзал", "Витамины", "Приём окулиста",
    ],
    "education": [
        "Онлайн-курс Python", "Книга по программированию",
        "Курс по ML", "Подписка Coursera", "Учебник",
    ],
    "gifts": [
        "Подарок на день рождения", "Цветы", "Подарок коллеге",
        "Подарок маме", "Сертификат в магазин",
    ],
    "electronics": [
        "Наушники", "Клавиатура механическая", "Монитор",
        "Веб-камера", "USB-хаб", "SSD-диск", "Мышь",
    ],
    "savings": ["Перевод на накопительный счёт"],
}

# --- Monthly patterns: (min_count, max_count, min_amount, max_amount) ---
EXPENSE_PATTERNS: dict[str, tuple[int, int, float, float]] = {
    "rent":          (1, 1, 45000, 45000),
    "groceries":     (20, 28, 350, 5500),
    "restaurant":    (8, 15, 450, 4000),
    "transport":     (22, 36, 50, 3500),
    "subscriptions": (2, 4, 199, 5990),
    "clothing":      (0, 3, 1500, 18000),
    "entertainment": (3, 8, 300, 5500),
    "health":        (1, 3, 400, 12000),
    "education":     (0, 2, 800, 12000),
    "gifts":         (0, 2, 500, 8000),
    "electronics":   (0, 1, 2000, 55000),
    "savings":       (1, 1, 10000, 35000),
}

INCOME_PATTERNS: dict[str, tuple[int, int, float, float]] = {
    "salary":   (1, 1, 180000, 195000),
    "freelance": (0, 3, 12000, 45000),
}

MONTH_NAMES_RU = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
]


def random_time() -> str:
    h = random.randint(7, 23)
    m = random.randint(0, 59)
    return f"{h:02d}:{m:02d}"


def random_day(year: int, month: int, day_hint: int | None = None) -> date:
    """Return a random date within the given month."""
    if month == 12:
        last_day = (date(year + 1, 1, 1) - timedelta(days=1)).day
    else:
        last_day = (date(year, month + 1, 1) - timedelta(days=1)).day
    if day_hint is not None:
        day = min(day_hint, last_day)
    else:
        day = random.randint(1, last_day)
    return date(year, month, day)


def pick_description(category: str, month: int) -> str:
    templates = DESCRIPTIONS[category]
    tpl = random.choice(templates)
    prev = (month - 2) % 12  # 0-indexed previous month
    return tpl.format(
        month_name=MONTH_NAMES_RU[month - 1],
        prev_month=MONTH_NAMES_RU[prev],
    )


def generate() -> list[dict]:
    transactions: list[dict] = []
    tx_id = 1

    # 12 months: April 2025 — March 2026
    months = [(2025, m) for m in range(4, 13)] + [(2026, m) for m in range(1, 4)]

    for year, month in months:
        # Income
        for category, (lo, hi, amin, amax) in INCOME_PATTERNS.items():
            count = random.randint(lo, hi)
            for _ in range(count):
                day_hint = 5 if category == "salary" else None
                d = random_day(year, month, day_hint)
                amount = round(random.uniform(amin, amax), 2)
                # Round salary to thousands
                if category == "salary":
                    amount = round(amount / 1000) * 1000
                transactions.append({
                    "id": tx_id,
                    "date": d.isoformat(),
                    "time": random_time(),
                    "type": "income",
                    "amount": float(amount),
                    "currency": "RUB",
                    "category": category,
                    "description": pick_description(category, month),
                })
                tx_id += 1

        # Expenses
        for category, (lo, hi, amin, amax) in EXPENSE_PATTERNS.items():
            count = random.randint(lo, hi)
            for _ in range(count):
                day_hint = 1 if category == "rent" else None
                d = random_day(year, month, day_hint)
                amount = round(random.uniform(amin, amax), 2)
                # Round rent to exact value
                if category == "rent":
                    amount = 45000.0
                transactions.append({
                    "id": tx_id,
                    "date": d.isoformat(),
                    "time": random_time(),
                    "type": "expense",
                    "amount": float(amount),
                    "currency": "RUB",
                    "category": category,
                    "description": pick_description(category, month),
                })
                tx_id += 1

    # Sort by date, then time
    transactions.sort(key=lambda t: (t["date"], t["time"]))

    # Re-assign sequential IDs after sorting
    for i, t in enumerate(transactions, 1):
        t["id"] = i

    return transactions


if __name__ == "__main__":
    data = generate()
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(data)} transactions → {OUTPUT}")
