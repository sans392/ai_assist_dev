"""Service for loading, validating, and querying health/sports data.

Loads data from three JSON files (activities, daily-facts, health-days),
validates via Pydantic models, flags anomalies, and provides query methods.

Stage 2 Level A: intent detection + pre-computed analytics.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
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

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Health_metrics_example_data"

ACTIVITIES_FILE = DATA_DIR / "activities.json"
DAILY_FACTS_FILE = DATA_DIR / "daily-facts.json"
HEALTH_DAYS_FILE = DATA_DIR / "health-days.json"

# ---------------------------------------------------------------------------
# Russian keyword stems → sport type names in data
# ---------------------------------------------------------------------------
_SPORT_KEYWORDS: dict[str, list[str]] = {
    "running": ["бег", "пробежк", "забег", "running", "run"],
    "cycling": ["велосипед", "велик", "велотренировк", "cycling", "bike"],
    "activity": ["активност", "activity", "общая"],
}

# Russian month name stems → month number
_MONTH_STEMS: list[tuple[str, str]] = [
    ("январ", "01"), ("феврал", "02"), ("март", "03"), ("марте", "03"),
    ("апрел", "04"), ("мая", "05"), ("май", "05"), ("мае", "05"),
    ("июн", "06"), ("июл", "07"), ("август", "08"),
    ("сентябр", "09"), ("октябр", "10"), ("ноябр", "11"), ("декабр", "12"),
]

_MONTH_NAMES_RU = {
    "01": "январь", "02": "февраль", "03": "март", "04": "апрель",
    "05": "май", "06": "июнь", "07": "июль", "08": "август",
    "09": "сентябрь", "10": "октябрь", "11": "ноябрь", "12": "декабрь",
}

_SPORT_NAMES_RU: dict[str, str] = {
    "running": "Бег",
    "cycling": "Велосипед",
    "activity": "Активность",
}

# Maximum number of individual activities to inject into the prompt
_ACTIVITY_LIMIT = 20


def _fmt_duration(seconds: int) -> str:
    """Format seconds as 'X ч Y мин' or 'X мин'."""
    if seconds < 60:
        return f"{seconds} сек"
    minutes = seconds // 60
    hours = minutes // 60
    mins = minutes % 60
    if hours > 0:
        return f"{hours} ч {mins} мин" if mins else f"{hours} ч"
    return f"{mins} мин"


def _fmt_distance(meters: float) -> str:
    """Format distance: km if >= 1000, otherwise metres."""
    if meters >= 1000:
        return f"{meters / 1000:.1f} км"
    return f"{meters:.0f} м"


def _month_label(ym: str) -> str:
    """'2026-03' → 'март 2026'."""
    parts = ym.split("-")
    name = _MONTH_NAMES_RU.get(parts[1], parts[1])
    return f"{name} {parts[0]}"


def _sport_ru(sport: str) -> str:
    return _SPORT_NAMES_RU.get(sport, sport)


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

        # Date range from daily facts
        dates = [f.isoDate for f in self._daily_facts]
        date_start = min(dates) if dates else None
        date_end = max(dates) if dates else None

        # Available metrics: check which fields actually have non-null values
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
            # Check sourcesJson for metrics that exist as keys
            for key in f.sources:
                available.add(key)

        # Sport types
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

    # ------------------------------------------------------------------
    # Public: summary and analytics
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Compact summary for system prompt."""
        if not self._daily_facts and not self._activities:
            return "Данные о здоровье и тренировках отсутствуют."

        report = self._load_report
        lines = ["--- СВОДКА ПО ДАННЫМ ---"]
        lines.append(f"Сегодняшняя дата: {self._today.isoformat()}")
        lines.append(f"Период данных: {report.date_range_start} — {report.date_range_end}")
        lines.append(f"Тренировок загружено: {report.activities_loaded}")
        lines.append(f"Виды спорта: {', '.join(report.sport_types)}")
        lines.append(f"Дней с метриками: {report.daily_facts_loaded}")
        lines.append(f"Доступные метрики: {', '.join(report.available_metrics)}")
        return "\n".join(lines)

    def get_analytics(self, user_message: str) -> str:
        """Analyse user question and return pre-computed answer as structured text."""
        if not self._daily_facts and not self._activities:
            return ""

        text = user_message.lower()
        sections: list[str] = []

        # --- Detect intent ---
        sport_types = self._detect_sport_types(text)
        date_range = self._detect_date_range(text)
        wants_full = self._wants_full_analysis(text)
        wants_recovery = self._wants_recovery(text)
        wants_steps = self._wants_steps(text)
        wants_calories = self._wants_calories(text)
        wants_activities = self._wants_activities(text)

        # --- Full analysis ---
        if wants_full:
            return self._build_full_analysis()

        # --- Activity list / workout summary ---
        if wants_activities or sport_types:
            sections.append(self._build_activity_summary(sport_types, date_range))

        # --- Recovery analysis ---
        if wants_recovery:
            sections.append(self._build_recovery_analysis(date_range))

        # --- Steps analysis ---
        if wants_steps:
            sections.append(self._build_steps_analysis(date_range))

        # --- Calories analysis ---
        if wants_calories and not wants_activities and not sport_types:
            sections.append(self._build_calories_analysis(date_range))

        # --- Date-range query without specific metric → daily metrics ---
        if date_range and not sections:
            sections.append(self._build_daily_metrics(date_range))

        # --- Fallback ---
        if not sections:
            sections.append(self._build_recent_summary())

        return "\n\n".join(s for s in sections if s)

    # ==================================================================
    # INTENT DETECTORS
    # ==================================================================

    def _detect_sport_types(self, text: str) -> set[str]:
        found: set[str] = set()
        for sport, keywords in _SPORT_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    found.add(sport)
                    break
        return found

    def _detect_date_range(self, text: str) -> tuple[date, date] | None:
        # "последние N дней"
        m = re.search(r"последн\w*\s+(\d+)\s*дн", text)
        if m:
            days = int(m.group(1))
            return (self._today - timedelta(days=days), self._today)

        # "за N дней"
        m = re.search(r"за\s+(\d+)\s*дн", text)
        if m:
            days = int(m.group(1))
            return (self._today - timedelta(days=days), self._today)

        # "прошлая/прошлую неделя/неделю"
        if "прошл" in text and "недел" in text:
            # Last Monday..Sunday
            today = self._today
            days_since_monday = today.weekday()
            last_monday = today - timedelta(days=days_since_monday + 7)
            last_sunday = last_monday + timedelta(days=6)
            return (last_monday, last_sunday)

        # "эта/текущая неделя"
        if any(kw in text for kw in ["эта недел", "этой недел", "текущ"]):
            today = self._today
            monday = today - timedelta(days=today.weekday())
            return (monday, today)

        # Russian month names
        months = self._detect_months(text)
        if months:
            ym = sorted(months)[0]
            year, month_num = int(ym[:4]), int(ym[5:7])
            first_day = date(year, month_num, 1)
            if month_num == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month_num + 1, 1) - timedelta(days=1)
            return (first_day, last_day)

        return None

    def _detect_months(self, text: str) -> set[str]:
        found: set[str] = set()

        # Direct YYYY-MM
        for m in re.findall(r"\d{4}-\d{2}", text):
            found.add(m)

        matched_nums: list[str] = []
        for stem, num in _MONTH_STEMS:
            if stem in text:
                matched_nums.append(num)

        is_last_year = "прошл" in text and ("год" in text or "лет" in text)
        current_year = str(self._today.year)
        previous_year = str(self._today.year - 1)

        # "прошлый месяц"
        if "прошл" in text and "месяц" in text and not matched_nums:
            prev = self._today.replace(day=1) - timedelta(days=1)
            found.add(f"{prev.year}-{prev.month:02d}")

        for num in matched_nums:
            if is_last_year:
                found.add(f"{previous_year}-{num}")
            else:
                # Check data availability
                all_dates = (
                    [f.isoDate.isoformat()[:7] for f in self._daily_facts]
                    + [a.startTime.strftime("%Y-%m") for a in self._activities]
                )
                target_ym = f"{current_year}-{num}"
                prev_ym = f"{previous_year}-{num}"
                if target_ym in all_dates:
                    found.add(target_ym)
                elif prev_ym in all_dates:
                    found.add(prev_ym)
                else:
                    found.add(target_ym)

        return found

    def _wants_full_analysis(self, text: str) -> bool:
        return any(
            kw in text
            for kw in [
                "полный анализ", "полная аналитика", "полный отчёт",
                "полный отчет", "общий анализ", "детальный анализ",
                "подробный анализ", "обзор всех данных",
            ]
        )

    def _wants_recovery(self, text: str) -> bool:
        return any(
            kw in text
            for kw in ["восстановлен", "recovery", "готовность"]
        )

    def _wants_steps(self, text: str) -> bool:
        return any(kw in text for kw in ["шаг", "steps", "ходьб"])

    def _wants_calories(self, text: str) -> bool:
        return any(kw in text for kw in ["калори", "calories", "ккал", "сжёг", "сжег"])

    def _wants_activities(self, text: str) -> bool:
        return any(
            kw in text
            for kw in [
                "тренировк", "трениров", "занят", "workout",
                "сколько трениров", "список тренировок",
            ]
        )

    # ==================================================================
    # ANALYTICS BUILDERS
    # ==================================================================

    def _filter_activities(
        self,
        sport_types: set[str] | None,
        date_range: tuple[date, date] | None,
    ) -> list[Activity]:
        result = self.clean_activities
        if sport_types:
            result = [a for a in result if a.sportType in sport_types]
        if date_range:
            d_from, d_to = date_range
            result = [
                a for a in result
                if d_from <= a.startTime.date() <= d_to
            ]
        return result

    def _filter_daily_facts(
        self, date_range: tuple[date, date] | None,
    ) -> list[DailyFact]:
        if not date_range:
            return list(self._daily_facts)
        d_from, d_to = date_range
        return [f for f in self._daily_facts if d_from <= f.isoDate <= d_to]

    def _build_activity_summary(
        self,
        sport_types: set[str] | None,
        date_range: tuple[date, date] | None,
    ) -> str:
        filtered = self._filter_activities(sport_types, date_range)

        # Header
        filter_parts: list[str] = []
        if sport_types:
            filter_parts.append(
                f"Виды: {', '.join(_sport_ru(s) for s in sorted(sport_types))}"
            )
        if date_range:
            filter_parts.append(f"Период: {date_range[0]} — {date_range[1]}")
        header = f" ({'; '.join(filter_parts)})" if filter_parts else ""

        if not filtered:
            return f"--- ТРЕНИРОВКИ{header} ---\nТренировки не найдены."

        total_duration = sum(a.duration for a in filtered)
        total_calories = sum(a.calories for a in filtered)
        total_distance = sum(a.distance for a in filtered)

        lines = [f"--- ТРЕНИРОВКИ{header} ---"]
        lines.append(f"Найдено тренировок: {len(filtered)}")
        lines.append(f"Общая длительность: {_fmt_duration(total_duration)}")
        lines.append(f"Общие калории: {total_calories:.0f} ккал")
        if total_distance > 0:
            lines.append(f"Общая дистанция: {_fmt_distance(total_distance)}")

        # Breakdown by sport type
        by_sport: dict[str, list[Activity]] = defaultdict(list)
        for a in filtered:
            by_sport[a.sportType].append(a)

        if len(by_sport) > 1:
            lines.append("")
            lines.append("По видам спорта:")
            for sport in sorted(by_sport):
                acts = by_sport[sport]
                dur = sum(a.duration for a in acts)
                cal = sum(a.calories for a in acts)
                lines.append(
                    f"  {_sport_ru(sport)}: {len(acts)} тренировок, "
                    f"{_fmt_duration(dur)}, {cal:.0f} ккал"
                )

        # List individual activities (most recent first, limited)
        recent = sorted(filtered, key=lambda a: a.startTime, reverse=True)
        show = recent[:_ACTIVITY_LIMIT]
        lines.append("")
        lines.append(f"Список тренировок (последние {len(show)}):")
        for a in show:
            dt = a.startTime.strftime("%Y-%m-%d %H:%M")
            parts = [
                f"  {dt}  {_sport_ru(a.sportType)}",
                _fmt_duration(a.duration),
                f"{a.calories:.0f} ккал",
            ]
            if a.distance > 0:
                parts.append(_fmt_distance(a.distance))
            lines.append("  |  ".join(parts))

        return "\n".join(lines)

    def _build_daily_metrics(
        self, date_range: tuple[date, date] | None,
    ) -> str:
        facts = self._filter_daily_facts(date_range)
        if not facts:
            period = f" ({date_range[0]} — {date_range[1]})" if date_range else ""
            return f"--- ДНЕВНЫЕ МЕТРИКИ{period} ---\nДанные не найдены."

        facts_sorted = sorted(facts, key=lambda f: f.isoDate)
        period = f" ({facts_sorted[0].isoDate} — {facts_sorted[-1].isoDate})"

        lines = [f"--- ДНЕВНЫЕ МЕТРИКИ{period} ---"]
        lines.append(f"Дней: {len(facts_sorted)}")

        # Steps
        steps_vals = [f.steps for f in facts_sorted if f.steps > 0]
        if steps_vals:
            lines.append("")
            lines.append("Шаги:")
            lines.append(f"  Среднее: {sum(steps_vals) // len(steps_vals)} шагов/день")
            lines.append(f"  Максимум: {max(steps_vals)} шагов")
            lines.append(f"  Минимум: {min(steps_vals)} шагов")
            lines.append(f"  Всего: {sum(steps_vals)} шагов за {len(steps_vals)} дней")

        # Calories
        cal_vals = [f.caloriesKcal for f in facts_sorted if f.caloriesKcal is not None]
        if cal_vals:
            lines.append("")
            lines.append("Калории:")
            lines.append(f"  Среднее: {sum(cal_vals) / len(cal_vals):.0f} ккал/день")
            lines.append(f"  Максимум: {max(cal_vals):.0f} ккал")
            lines.append(f"  Минимум: {min(cal_vals):.0f} ккал")

        # Recovery
        rec_vals = [f.recoveryScore for f in facts_sorted if f.recoveryScore is not None]
        if rec_vals:
            lines.append("")
            lines.append("Восстановление (recovery score):")
            lines.append(f"  Среднее: {sum(rec_vals) / len(rec_vals):.0f}%")
            lines.append(f"  Максимум: {max(rec_vals):.0f}%")
            lines.append(f"  Минимум: {min(rec_vals):.0f}%")

        # Sleep
        sleep_vals = [
            f.sleepPerformancePercentage
            for f in facts_sorted
            if f.sleepPerformancePercentage is not None
        ]
        if sleep_vals:
            lines.append("")
            lines.append("Качество сна:")
            lines.append(f"  Среднее: {sum(sleep_vals) / len(sleep_vals):.0f}%")
            lines.append(f"  Максимум: {max(sleep_vals):.0f}%")
            lines.append(f"  Минимум: {min(sleep_vals):.0f}%")

        return "\n".join(lines)

    def _build_recovery_analysis(
        self, date_range: tuple[date, date] | None,
    ) -> str:
        facts = self._filter_daily_facts(date_range)
        with_recovery = [
            f for f in sorted(facts, key=lambda f: f.isoDate)
            if f.recoveryScore is not None
        ]

        if not with_recovery:
            return "--- АНАЛИЗ ВОССТАНОВЛЕНИЯ ---\nДанные о recovery score отсутствуют."

        vals = [f.recoveryScore for f in with_recovery]
        avg = sum(vals) / len(vals)

        lines = ["--- АНАЛИЗ ВОССТАНОВЛЕНИЯ ---"]
        lines.append(f"Дней с данными: {len(with_recovery)}")
        lines.append(f"Средний recovery score: {avg:.0f}%")
        lines.append(f"Максимум: {max(vals):.0f}%")
        lines.append(f"Минимум: {min(vals):.0f}%")

        # Categorize days
        low = sum(1 for v in vals if v < 34)
        medium = sum(1 for v in vals if 34 <= v < 67)
        high = sum(1 for v in vals if v >= 67)
        lines.append("")
        lines.append("Распределение:")
        lines.append(f"  Низкое (< 34%): {low} дней")
        lines.append(f"  Среднее (34–66%): {medium} дней")
        lines.append(f"  Высокое (≥ 67%): {high} дней")

        # Daily breakdown
        lines.append("")
        lines.append("По дням:")
        for f in with_recovery:
            lines.append(f"  {f.isoDate}: {f.recoveryScore:.0f}%")

        return "\n".join(lines)

    def _build_steps_analysis(
        self, date_range: tuple[date, date] | None,
    ) -> str:
        facts = self._filter_daily_facts(date_range)
        with_steps = [
            f for f in sorted(facts, key=lambda f: f.isoDate)
            if f.steps > 0
        ]

        if not with_steps:
            return "--- АНАЛИЗ ШАГОВ ---\nДанные о шагах отсутствуют."

        vals = [f.steps for f in with_steps]
        avg = sum(vals) // len(vals)
        total = sum(vals)

        lines = ["--- АНАЛИЗ ШАГОВ ---"]
        lines.append(f"Дней с данными: {len(with_steps)}")
        lines.append(f"Всего шагов: {total}")
        lines.append(f"Среднее: {avg} шагов/день")
        lines.append(f"Максимум: {max(vals)} шагов")
        lines.append(f"Минимум: {min(vals)} шагов")

        # Days above/below average
        above = sum(1 for v in vals if v >= avg)
        below = len(vals) - above
        lines.append("")
        lines.append(f"Дней выше среднего: {above}")
        lines.append(f"Дней ниже среднего: {below}")

        # Daily breakdown
        lines.append("")
        lines.append("По дням:")
        for f in with_steps:
            lines.append(f"  {f.isoDate}: {f.steps} шагов")

        return "\n".join(lines)

    def _build_calories_analysis(
        self, date_range: tuple[date, date] | None,
    ) -> str:
        facts = self._filter_daily_facts(date_range)
        with_cal = [
            f for f in sorted(facts, key=lambda f: f.isoDate)
            if f.caloriesKcal is not None
        ]

        if not with_cal:
            return "--- АНАЛИЗ КАЛОРИЙ ---\nДанные о калориях отсутствуют."

        vals = [f.caloriesKcal for f in with_cal]
        avg = sum(vals) / len(vals)

        lines = ["--- АНАЛИЗ КАЛОРИЙ ---"]
        lines.append(f"Дней с данными: {len(with_cal)}")
        lines.append(f"Среднее: {avg:.0f} ккал/день")
        lines.append(f"Максимум: {max(vals):.0f} ккал")
        lines.append(f"Минимум: {min(vals):.0f} ккал")
        lines.append(f"Всего: {sum(vals):.0f} ккал за {len(with_cal)} дней")

        # Daily breakdown
        lines.append("")
        lines.append("По дням:")
        for f in with_cal:
            lines.append(f"  {f.isoDate}: {f.caloriesKcal:.0f} ккал")

        return "\n".join(lines)

    def _build_sport_breakdown(self) -> str:
        """Breakdown of activities by sport type across all data."""
        activities = self.clean_activities
        if not activities:
            return "--- РАЗБИВКА ПО ВИДАМ СПОРТА ---\nТренировки отсутствуют."

        by_sport: dict[str, list[Activity]] = defaultdict(list)
        for a in activities:
            by_sport[a.sportType].append(a)

        lines = ["--- РАЗБИВКА ПО ВИДАМ СПОРТА ---"]
        lines.append(f"Всего тренировок: {len(activities)}")
        lines.append("")

        for sport in sorted(by_sport, key=lambda s: -len(by_sport[s])):
            acts = by_sport[sport]
            total_dur = sum(a.duration for a in acts)
            total_cal = sum(a.calories for a in acts)
            total_dist = sum(a.distance for a in acts)
            avg_dur = total_dur // len(acts)

            lines.append(f"{_sport_ru(sport)}:")
            lines.append(f"  Тренировок: {len(acts)}")
            lines.append(f"  Общая длительность: {_fmt_duration(total_dur)}")
            lines.append(f"  Средняя длительность: {_fmt_duration(avg_dur)}")
            lines.append(f"  Общие калории: {total_cal:.0f} ккал")
            if total_dist > 0:
                lines.append(f"  Общая дистанция: {_fmt_distance(total_dist)}")
            lines.append("")

        return "\n".join(lines)

    def _build_full_analysis(self) -> str:
        """Comprehensive pre-computed analysis of all health/sports data."""
        sections = [
            self.get_summary(),
            self._build_sport_breakdown(),
            self._build_daily_metrics(None),
            self._build_recovery_analysis(None),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_recent_summary(self) -> str:
        """Fallback: summary of recent data (last 7 days)."""
        d_to = self._today
        d_from = d_to - timedelta(days=7)
        date_range = (d_from, d_to)

        activities = self._filter_activities(None, date_range)
        facts = self._filter_daily_facts(date_range)

        lines = [f"--- ПОСЛЕДНИЕ 7 ДНЕЙ ({d_from} — {d_to}) ---"]

        if activities:
            total_dur = sum(a.duration for a in activities)
            total_cal = sum(a.calories for a in activities)
            lines.append(f"Тренировок: {len(activities)}")
            lines.append(f"Общая длительность: {_fmt_duration(total_dur)}")
            lines.append(f"Калории (тренировки): {total_cal:.0f} ккал")
        else:
            lines.append("Тренировок не найдено.")

        steps_vals = [f.steps for f in facts if f.steps > 0]
        if steps_vals:
            lines.append(f"Среднее шагов/день: {sum(steps_vals) // len(steps_vals)}")

        rec_vals = [f.recoveryScore for f in facts if f.recoveryScore is not None]
        if rec_vals:
            lines.append(f"Средний recovery: {sum(rec_vals) / len(rec_vals):.0f}%")

        if not activities and not facts:
            lines.append("Данные за этот период отсутствуют.")
            # Fall back to showing all-time summary
            lines.append("")
            lines.append(self._build_sport_breakdown())

        return "\n".join(lines)


# Module-level singleton
sports_service = SportsService()
