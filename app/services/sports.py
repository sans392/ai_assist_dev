"""Service for loading, validating, and querying health/sports data.

Loads data from three JSON files (activities, daily-facts, health-days),
validates via Pydantic models, flags anomalies, and provides query methods.

Stage 2 Level A: intent detection + pre-computed analytics for reliable metrics.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from pydantic import ValidationError

from app.models.health_schemas import (
    Activity,
    ActivityFlag,
    DataLoadReport,
    DailyFact,
    HealthDay,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Health_metrics_example_data"

ACTIVITIES_FILE = DATA_DIR / "activities.json"
DAILY_FACTS_FILE = DATA_DIR / "daily-facts.json"
HEALTH_DAYS_FILE = DATA_DIR / "health-days.json"

# ---------------------------------------------------------------------------
# Russian month stems -> month number
# ---------------------------------------------------------------------------
_MONTH_STEMS: list[tuple[str, int]] = [
    ("январ", 1), ("феврал", 2), ("март", 3), ("марте", 3),
    ("апрел", 4), ("мая", 5), ("май", 5), ("мае", 5),
    ("июн", 6), ("июл", 7), ("август", 8),
    ("сентябр", 9), ("октябр", 10), ("ноябр", 11), ("декабр", 12),
]

_MONTH_NAMES_RU = {
    1: "январь", 2: "февраль", 3: "март", 4: "апрель",
    5: "май", 6: "июнь", 7: "июль", 8: "август",
    9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь",
}

# Sport type keywords -> sportType values in data
_SPORT_KEYWORDS: dict[str, list[str]] = {
    "running": ["бег", "пробежк", "running", "пробег"],
    "cycling": ["велосипед", "cycling", "велотренировк", "велик"],
}


def _fmt_duration(seconds: int) -> str:
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h} ч {m} мин" if m else f"{h} ч"
    return f"{seconds // 60} мин"


def _fmt_distance(meters: float) -> str:
    if meters >= 1000:
        return f"{meters / 1000:.1f} км"
    if meters > 0:
        return f"{meters:.0f} м"
    return "—"


def _fmt_steps(n: int) -> str:
    return f"{n:,}".replace(",", " ")


class SportsService:
    def __init__(self) -> None:
        self._activities: list[Activity] = []
        self._daily_facts: list[DailyFact] = []
        self._health_days: list[HealthDay] = []
        self._load_report = DataLoadReport()
        self._today: date = date.today()
        self._load()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._activities = self._load_activities()
        self._daily_facts = self._load_daily_facts()
        self._health_days = self._load_health_days()
        self._load_report = self._build_report()

        logger.info(
            "SportsService loaded: %d activities (%d flagged), "
            "%d daily facts, %d health days",
            len(self._activities),
            sum(1 for a in self._activities if a.flags),
            len(self._daily_facts),
            len(self._health_days),
        )

    def _load_json_data(self, path: Path) -> list[dict]:
        """Load JSON file and return the 'data' array."""
        if not path.exists():
            logger.warning("Data file not found: %s", path)
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "data" in raw:
                return raw["data"]
            if isinstance(raw, list):
                return raw
            logger.warning("Unexpected JSON structure in %s", path)
            return []
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read %s: %s", path, e)
            return []

    def _load_activities(self) -> list[Activity]:
        raw_list = self._load_json_data(ACTIVITIES_FILE)
        activities: list[Activity] = []
        skipped = 0
        for raw in raw_list:
            try:
                activities.append(Activity.model_validate(raw))
            except ValidationError as e:
                skipped += 1
                logger.warning(
                    "Skipping invalid activity %s: %s",
                    raw.get("id", "unknown"),
                    e,
                )
        if skipped:
            logger.warning("Skipped %d invalid activities", skipped)
        self._activities_skipped = skipped
        return activities

    def _load_daily_facts(self) -> list[DailyFact]:
        raw_list = self._load_json_data(DAILY_FACTS_FILE)
        facts: list[DailyFact] = []
        skipped = 0
        for raw in raw_list:
            try:
                facts.append(DailyFact.model_validate(raw))
            except ValidationError as e:
                skipped += 1
                logger.warning(
                    "Skipping invalid daily fact %s: %s",
                    raw.get("id", "unknown"),
                    e,
                )
        if skipped:
            logger.warning("Skipped %d invalid daily facts", skipped)
        self._daily_facts_skipped = skipped
        return facts

    def _load_health_days(self) -> list[HealthDay]:
        raw_list = self._load_json_data(HEALTH_DAYS_FILE)
        days: list[HealthDay] = []
        skipped = 0
        for raw in raw_list:
            try:
                days.append(HealthDay.model_validate(raw))
            except ValidationError as e:
                skipped += 1
                logger.warning(
                    "Skipping invalid health day %s: %s",
                    raw.get("id", "unknown"),
                    e,
                )
        if skipped:
            logger.warning("Skipped %d invalid health days", skipped)
        self._health_days_skipped = skipped
        return days

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_report(self) -> DataLoadReport:
        """Build a summary report of loaded data."""
        activities_flagged = sum(1 for a in self._activities if a.flags)

        dates = [f.isoDate for f in self._daily_facts]
        date_start = min(dates) if dates else None
        date_end = max(dates) if dates else None

        available: set[str] = set()
        for f in self._daily_facts:
            if f.steps > 0:
                available.add("steps")
            if f.caloriesKcal is not None:
                available.add("caloriesKcal")
            if f.recoveryScore is not None:
                available.add("recoveryScore")
            if f.sleepPerformancePercentage is not None:
                available.add("sleepPerformancePercentage")
            for key in f.sources:
                available.add(key)

        sport_types = sorted({a.sportType for a in self._activities})

        return DataLoadReport(
            activities_loaded=len(self._activities),
            activities_skipped=getattr(self, "_activities_skipped", 0),
            activities_flagged=activities_flagged,
            daily_facts_loaded=len(self._daily_facts),
            daily_facts_skipped=getattr(self, "_daily_facts_skipped", 0),
            health_days_loaded=len(self._health_days),
            health_days_skipped=getattr(self, "_health_days_skipped", 0),
            date_range_start=date_start,
            date_range_end=date_end,
            available_metrics=sorted(available),
            sport_types=sport_types,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def activities(self) -> list[Activity]:
        return self._activities

    @property
    def clean_activities(self) -> list[Activity]:
        """Activities with test/debug data excluded."""
        return [a for a in self._activities if ActivityFlag.TEST_DATA not in a.flags]

    @property
    def daily_facts(self) -> list[DailyFact]:
        return self._daily_facts

    @property
    def health_days(self) -> list[HealthDay]:
        return self._health_days

    @property
    def load_report(self) -> DataLoadReport:
        return self._load_report

    # ==================================================================
    # PUBLIC: summary (always in system prompt)
    # ==================================================================

    def get_summary(self) -> str:
        """Compact summary with pre-computed aggregates for system prompt."""
        if not self._daily_facts and not self._activities:
            return "Данные о здоровье и тренировках отсутствуют."

        report = self._load_report
        lines = [
            "=== СВОДКА ПО ЗДОРОВЬЮ И ТРЕНИРОВКАМ ===",
            f"Сегодняшняя дата: {self._today.isoformat()}",
            f"Период данных: {report.date_range_start} — {report.date_range_end}",
        ]

        # --- Steps for last 7 days ---
        d7 = self._today - timedelta(days=7)
        facts_7d = [f for f in self._daily_facts if d7 <= f.isoDate <= self._today]
        if facts_7d:
            total_steps = sum(f.steps for f in facts_7d)
            avg_steps = total_steps // len(facts_7d)
            lines.append("")
            lines.append(
                f"Шаги за последние 7 дней: "
                f"{_fmt_steps(total_steps)} всего, "
                f"{_fmt_steps(avg_steps)} в среднем/день"
            )
        else:
            lines.append("")
            lines.append("Шаги за последние 7 дней: нет данных")

        # --- Training count by sport type (last 7 / 30 days) ---
        clean = self.clean_activities
        d30 = self._today - timedelta(days=30)

        act_7d = [a for a in clean if d7 <= a.startTime.date() <= self._today]
        act_30d = [a for a in clean if d30 <= a.startTime.date() <= self._today]

        lines.append("")
        if act_7d:
            by_sport = Counter(a.sportType for a in act_7d)
            parts = [f"{s}: {c}" for s, c in sorted(by_sport.items())]
            lines.append(f"Тренировки за 7 дней: {len(act_7d)} ({', '.join(parts)})")
        else:
            lines.append("Тренировки за 7 дней: нет данных")

        if act_30d:
            by_sport = Counter(a.sportType for a in act_30d)
            parts = [f"{s}: {c}" for s, c in sorted(by_sport.items())]
            lines.append(f"Тренировки за 30 дней: {len(act_30d)} ({', '.join(parts)})")
        else:
            lines.append("Тренировки за 30 дней: нет данных")

        # --- Recovery score + trend ---
        recovery_pairs = sorted(
            [(f.isoDate, f.recoveryScore) for f in self._daily_facts
             if f.recoveryScore is not None],
            key=lambda x: x[0],
        )
        if recovery_pairs:
            latest_date, latest_score = recovery_pairs[-1]
            lines.append("")
            lines.append(
                f"Восстановление (последний): {latest_score:.0f}% ({latest_date})"
            )
            last_5 = [v for _, v in recovery_pairs[-5:]]
            if len(last_5) >= 2:
                lines.append(
                    f"Тренд (последние {len(last_5)} значений): "
                    f"{_compute_trend(last_5)}"
                )
        else:
            lines.append("")
            lines.append("Восстановление: нет данных")

        return "\n".join(lines)

    # ==================================================================
    # PUBLIC: context-aware analytics
    # ==================================================================

    def get_analytics(self, user_message: str) -> str:
        """Analyse user question and return pre-computed analytics."""
        if not user_message:
            return ""
        if not self._daily_facts and not self._activities:
            return ""

        text = user_message.lower()
        sections: list[str] = []

        # --- Detect intents ---
        date_range = self._detect_date_range(text)
        sport_types = self._detect_sport_type(text)
        wants_steps = self._detect_steps_intent(text)
        wants_training = self._detect_training_intent(text)
        wants_recovery = self._detect_recovery_intent(text)

        d_from, d_to = date_range if date_range else self._default_date_range()

        # --- Build sections ---
        if wants_steps:
            sections.append(self._build_steps_analytics(d_from, d_to))

        if wants_training:
            sections.append(self._build_training_list(d_from, d_to, sport_types))

        if wants_recovery:
            sections.append(self._build_recovery_analytics(d_from, d_to))

        if sport_types and not wants_training:
            sections.append(self._build_sport_breakdown(sport_types, d_from, d_to))

        # Fallback
        if not sections:
            sections.append(self._build_recent_summary())

        return "\n\n".join(s for s in sections if s)

    # ==================================================================
    # ANALYTICS BUILDERS
    # ==================================================================

    def _build_steps_analytics(self, d_from: date, d_to: date) -> str:
        facts = sorted(
            [f for f in self._daily_facts if d_from <= f.isoDate <= d_to],
            key=lambda f: f.isoDate,
        )

        lines = [f"--- ШАГИ ({d_from} — {d_to}) ---"]
        if not facts:
            lines.append("Нет данных о шагах за указанный период.")
            return "\n".join(lines)

        total = sum(f.steps for f in facts)
        avg = total // len(facts)

        lines.append(f"Всего: {_fmt_steps(total)} шагов за {len(facts)} дней")
        lines.append(f"В среднем: {_fmt_steps(avg)} шагов/день")
        lines.append("")
        lines.append("По дням:")
        for f in facts:
            lines.append(f"  {f.isoDate}: {_fmt_steps(f.steps)} шагов")

        return "\n".join(lines)

    def _build_training_list(
        self, d_from: date, d_to: date, sport_types: set[str] | None,
    ) -> str:
        activities = sorted(
            [
                a for a in self.clean_activities
                if d_from <= a.startTime.date() <= d_to
                and (not sport_types or a.sportType in sport_types)
            ],
            key=lambda a: a.startTime,
        )

        filter_desc = f"{d_from} — {d_to}"
        if sport_types:
            filter_desc += f", {', '.join(sorted(sport_types))}"

        lines = [f"--- ТРЕНИРОВКИ ({filter_desc}) ---"]
        if not activities:
            lines.append("Тренировок не найдено за указанный период.")
            return "\n".join(lines)

        lines.append(f"Найдено: {len(activities)}")
        total_duration = sum(a.duration for a in activities)
        total_calories = sum(a.calories for a in activities)
        lines.append(f"Общая длительность: {_fmt_duration(total_duration)}")
        lines.append(f"Общие калории: {total_calories:.0f} kcal")
        lines.append("")

        for a in activities:
            d_str = a.startTime.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  {d_str} | {a.title} | {a.sportType} | "
                f"{_fmt_duration(a.duration)} | {_fmt_distance(a.distance)} | "
                f"{a.calories:.0f} kcal"
            )

        return "\n".join(lines)

    def _build_recovery_analytics(self, d_from: date, d_to: date) -> str:
        facts = sorted(
            [f for f in self._daily_facts
             if d_from <= f.isoDate <= d_to and f.recoveryScore is not None],
            key=lambda f: f.isoDate,
        )

        lines = [f"--- ВОССТАНОВЛЕНИЕ ({d_from} — {d_to}) ---"]
        if not facts:
            lines.append("Нет данных о восстановлении за указанный период.")
            return "\n".join(lines)

        scores = [f.recoveryScore for f in facts]
        avg_score = sum(scores) / len(scores)

        lines.append(f"Записей: {len(facts)}")
        lines.append(f"Среднее: {avg_score:.0f}%")
        lines.append(f"Минимум: {min(scores):.0f}%")
        lines.append(f"Максимум: {max(scores):.0f}%")

        if len(scores) >= 2:
            lines.append(f"Тренд: {_compute_trend(scores)}")

        lines.append("")
        lines.append("По дням:")
        for f in facts:
            lines.append(f"  {f.isoDate}: {f.recoveryScore:.0f}%")

        return "\n".join(lines)

    def _build_sport_breakdown(
        self, sport_types: set[str], d_from: date, d_to: date,
    ) -> str:
        activities = sorted(
            [a for a in self.clean_activities
             if d_from <= a.startTime.date() <= d_to and a.sportType in sport_types],
            key=lambda a: a.startTime,
        )

        sport_label = ", ".join(sorted(sport_types))
        lines = [f"--- {sport_label.upper()} ({d_from} — {d_to}) ---"]

        if not activities:
            lines.append("Тренировок данного типа не найдено.")
            return "\n".join(lines)

        lines.append(f"Всего тренировок: {len(activities)}")
        total_duration = sum(a.duration for a in activities)
        total_calories = sum(a.calories for a in activities)
        total_distance = sum(a.distance for a in activities)
        avg_duration = total_duration // len(activities)

        lines.append(f"Общая длительность: {_fmt_duration(total_duration)}")
        lines.append(f"Средняя длительность: {_fmt_duration(avg_duration)}")
        lines.append(f"Общая дистанция: {_fmt_distance(total_distance)}")
        lines.append(f"Общие калории: {total_calories:.0f} kcal")
        lines.append("")

        for a in activities:
            d_str = a.startTime.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  {d_str} | {a.title} | {_fmt_duration(a.duration)} | "
                f"{_fmt_distance(a.distance)} | {a.calories:.0f} kcal"
            )

        return "\n".join(lines)

    def _build_recent_summary(self) -> str:
        """Fallback: summary of the last week of available data."""
        dates = [f.isoDate for f in self._daily_facts]
        if not dates:
            return ""

        d_to = max(dates)
        d_from = d_to - timedelta(days=7)

        facts = sorted(
            [f for f in self._daily_facts if d_from <= f.isoDate <= d_to],
            key=lambda f: f.isoDate,
        )
        lines = [f"--- ОБЗОР ({d_from} — {d_to}) ---"]
        if facts:
            total = sum(f.steps for f in facts)
            avg = total // len(facts)
            lines.append(
                f"Шаги: {_fmt_steps(total)} всего, "
                f"{_fmt_steps(avg)} в среднем/день"
            )

        clean = self.clean_activities
        recent_acts = [a for a in clean if d_from <= a.startTime.date() <= d_to]
        if recent_acts:
            by_sport = Counter(a.sportType for a in recent_acts)
            parts = [f"{s}: {c}" for s, c in sorted(by_sport.items())]
            lines.append(f"Тренировки: {len(recent_acts)} ({', '.join(parts)})")

        return "\n".join(lines)

    # ==================================================================
    # INTENT DETECTORS
    # ==================================================================

    def _detect_steps_intent(self, text: str) -> bool:
        return any(kw in text for kw in ["шаги", "шагов", "ходьб"])

    def _detect_training_intent(self, text: str) -> bool:
        return any(kw in text for kw in [
            "тренировк", "тренировок", "заняти", "активност",
        ])

    def _detect_recovery_intent(self, text: str) -> bool:
        return any(kw in text for kw in ["восстановлен", "recovery"])

    def _detect_sport_type(self, text: str) -> set[str] | None:
        found: set[str] = set()
        for sport, keywords in _SPORT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                found.add(sport)
        return found or None

    def _detect_date_range(self, text: str) -> tuple[date, date] | None:
        today = self._today

        # "вчера"
        if "вчера" in text:
            yesterday = today - timedelta(days=1)
            return (yesterday, yesterday)

        # "сегодня"
        if "сегодня" in text:
            return (today, today)

        # "на прошлой неделе" / "прошлая неделя"
        if "прошл" in text and "недел" in text:
            days_since_monday = today.weekday()
            this_monday = today - timedelta(days=days_since_monday)
            last_monday = this_monday - timedelta(days=7)
            last_sunday = this_monday - timedelta(days=1)
            return (last_monday, last_sunday)

        # "последние N дней"
        m = re.search(r"последн\w*\s+(\d+)\s*дн", text)
        if m:
            days = int(m.group(1))
            return (today - timedelta(days=days), today)

        # "за N дней"
        m = re.search(r"за\s+(\d+)\s*дн", text)
        if m:
            days = int(m.group(1))
            return (today - timedelta(days=days), today)

        # Month names: "за март", "в марте", etc.
        for stem, month_num in _MONTH_STEMS:
            if stem in text:
                year = today.year
                if month_num > today.month:
                    year -= 1
                if "прошл" in text and ("год" in text or "лет" in text):
                    year -= 1

                d_from = date(year, month_num, 1)
                if month_num == 12:
                    d_to = date(year + 1, 1, 1) - timedelta(days=1)
                else:
                    d_to = date(year, month_num + 1, 1) - timedelta(days=1)
                return (d_from, d_to)

        return None

    def _default_date_range(self) -> tuple[date, date]:
        return (self._today - timedelta(days=30), self._today)


# ==================================================================
# MODULE HELPERS
# ==================================================================

def _compute_trend(values: list[float]) -> str:
    if len(values) < 2:
        return "недостаточно данных"
    diff = values[-1] - values[0]
    if diff > 5:
        return "↑ растёт"
    if diff < -5:
        return "↓ падает"
    return "→ стабильно"


# Module-level singleton
sports_service = SportsService()
