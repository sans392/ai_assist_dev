"""Service for loading, validating, and querying health/sports data.

Loads data from three JSON files (activities, daily-facts, health-days),
validates via Pydantic models, flags anomalies, and provides query methods.

Stage 2 Level A: intent detection + pre-computed analytics for reliable metrics.
Stage 3 Level B: deduplication across sources + cross-file analytics
(calories fallback, rest days, weekly load, running progress by pace).
Stage 4 Level C: metric availability detection (graceful degradation when
HRV/sleep/SpO2/resting HR/skin temp data is missing) + correlations
(recovery vs training load, sleep vs recovery, HRV vs training intensity).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
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

# Source priority for selecting canonical record among duplicates.
# Higher number = higher priority.
_SOURCE_PRIORITY: dict[str, int] = {
    "whoop": 3,
    "apple_health": 2,
    "manual": 1,
}

# Threshold for treating two activities as duplicates:
# overlap duration / min(duration_a, duration_b) > this value.
_DUPLICATE_OVERLAP_RATIO = 0.5

# Canonical metric keys tracked for availability. Order matters for stable
# display. Each key maps to a human-readable Russian label used in system
# prompts and "нет данных" messages.
_METRIC_LABELS_RU: dict[str, str] = {
    "steps": "шаги",
    "caloriesKcal": "калории",
    "recoveryScore": "восстановление",
    "sleepPerformancePercentage": "эффективность сна",
    "hrvRmssdMilli": "HRV (вариабельность пульса)",
    "restingHeartRate": "пульс в покое",
    "spo2Percentage": "SpO2 (насыщение кислородом)",
    "skinTempCelsius": "температура кожи",
    "sleepTotalInBedTimeMilli": "длительность сна",
}


def _activity_source(a: Activity) -> str:
    """Infer the recording source of an activity from its title."""
    title = (a.title or "").lower()
    if "whoop" in title:
        return "whoop"
    if "apple" in title:
        return "apple_health"
    return "manual"


def _overlap_seconds(a: Activity, b: Activity) -> float:
    """Return the number of seconds that two activities overlap in time."""
    start = max(a.startTime, b.startTime)
    end = min(a.endTime, b.endTime)
    delta = (end - start).total_seconds()
    return max(0.0, delta)


def _are_duplicates(a: Activity, b: Activity) -> bool:
    """Two activities are duplicates if they are the same day, same sport,
    and overlap for more than DUPLICATE_OVERLAP_RATIO of the shorter one."""
    if a.sportType != b.sportType:
        return False
    if a.startTime.date() != b.startTime.date():
        return False
    overlap = _overlap_seconds(a, b)
    if overlap <= 0:
        return False
    min_dur = min(a.duration, b.duration)
    if min_dur <= 0:
        return False
    return (overlap / min_dur) > _DUPLICATE_OVERLAP_RATIO


def _pick_canonical(group: list[Activity]) -> Activity:
    """Pick the canonical record from a group of overlapping activities.

    Priority: source priority (WHOOP > Apple Health > manual),
    tiebroken by longer duration, then by earliest start time.
    """
    return max(
        group,
        key=lambda a: (
            _SOURCE_PRIORITY.get(_activity_source(a), 0),
            a.duration,
            -a.startTime.timestamp(),
        ),
    )


def _deduplicate(activities: list[Activity]) -> tuple[list[Activity], int]:
    """Deduplicate a list of activities by time-overlap + source priority.

    Returns (deduped_activities, removed_count).
    The input is assumed to already be filtered of test/debug records.
    """
    if not activities:
        return [], 0

    # Group by (day, sportType) to limit pairwise comparisons
    by_bucket: dict[tuple[date, str], list[int]] = defaultdict(list)
    for idx, a in enumerate(activities):
        by_bucket[(a.startTime.date(), a.sportType)].append(idx)

    # Union-find across the full list
    parent = list(range(len(activities)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for indices in by_bucket.values():
        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                if _are_duplicates(activities[indices[i]], activities[indices[j]]):
                    union(indices[i], indices[j])

    # Build groups, pick canonical per group
    groups: dict[int, list[Activity]] = defaultdict(list)
    for idx, a in enumerate(activities):
        groups[find(idx)].append(a)

    deduped = [_pick_canonical(g) for g in groups.values()]
    deduped.sort(key=lambda a: a.startTime)
    removed = len(activities) - len(deduped)
    return deduped, removed


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
        self._unique_activities: list[Activity] = []
        self._activities_duplicates_removed: int = 0
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

        # Stage 3: deduplicate clean activities across sources
        clean = [
            a for a in self._activities
            if ActivityFlag.TEST_DATA not in a.flags
        ]
        self._unique_activities, self._activities_duplicates_removed = _deduplicate(clean)

        self._load_report = self._build_report()

        logger.info(
            "SportsService loaded: %d activities (%d flagged, %d duplicates), "
            "%d unique, %d daily facts, %d health days",
            len(self._activities),
            sum(1 for a in self._activities if a.flags),
            self._activities_duplicates_removed,
            len(self._unique_activities),
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

        available = self._compute_available_metrics()
        unavailable = [m for m in _METRIC_LABELS_RU if m not in available]

        sport_types = sorted({a.sportType for a in self._activities})

        return DataLoadReport(
            activities_loaded=len(self._activities),
            activities_skipped=getattr(self, "_activities_skipped", 0),
            activities_flagged=activities_flagged,
            activities_unique=len(self._unique_activities),
            activities_duplicates_removed=self._activities_duplicates_removed,
            daily_facts_loaded=len(self._daily_facts),
            daily_facts_skipped=getattr(self, "_daily_facts_skipped", 0),
            health_days_loaded=len(self._health_days),
            health_days_skipped=getattr(self, "_health_days_skipped", 0),
            date_range_start=date_start,
            date_range_end=date_end,
            available_metrics=available,
            unavailable_metrics=unavailable,
            sport_types=sport_types,
        )

    def _compute_available_metrics(self) -> list[str]:
        """Determine which canonical metrics have at least one real value.

        Availability is judged by actual stored field values on DailyFact
        records — NOT by keys mentioned in `sourcesJson`. A metric is
        "available" only if at least one daily-fact has a non-null value
        for that metric field.
        """
        available: list[str] = []
        for metric in _METRIC_LABELS_RU:
            if self._metric_has_any_value(metric):
                available.append(metric)
        return available

    def _metric_has_any_value(self, metric: str) -> bool:
        """Check if any daily fact has a real value for the given metric."""
        if metric == "steps":
            return any(f.steps > 0 for f in self._daily_facts)
        for f in self._daily_facts:
            if getattr(f, metric, None) is not None:
                return True
        return False

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
    def unique_activities(self) -> list[Activity]:
        """Deduplicated clean activities across sources.

        Duplicates (same day, same sport type, >50% time overlap) are
        collapsed to a single canonical record (source priority: WHOOP >
        Apple Health > manual). Test/debug records are excluded.
        """
        return self._unique_activities

    @property
    def duplicates_removed(self) -> int:
        """Number of activity records removed by deduplication."""
        return self._activities_duplicates_removed

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
    # Stage 4: metric availability (graceful degradation)
    # ------------------------------------------------------------------

    def get_available_metrics(self) -> list[str]:
        """Canonical metric names that have at least one real value in data."""
        return list(self._load_report.available_metrics)

    def get_unavailable_metrics(self) -> list[str]:
        """Canonical metric names tracked but absent from data."""
        return list(self._load_report.unavailable_metrics)

    def is_metric_available(self, metric: str) -> bool:
        """Whether the given canonical metric has any data."""
        return metric in self._load_report.available_metrics

    def get_availability_section(self) -> str:
        """Human-readable availability block for the system prompt."""
        available = self.get_available_metrics()
        unavailable = self.get_unavailable_metrics()

        available_labels = [_METRIC_LABELS_RU[m] for m in available]
        unavailable_labels = [_METRIC_LABELS_RU[m] for m in unavailable]

        lines = ["--- ДОСТУПНОСТЬ МЕТРИК ---"]
        if available_labels:
            lines.append(f"Доступные метрики: {', '.join(available_labels)}")
        else:
            lines.append("Доступные метрики: нет")
        if unavailable_labels:
            lines.append(
                f"Недоступные метрики: {', '.join(unavailable_labels)}"
            )
            lines.append(
                "ВАЖНО: для недоступных метрик нет данных — "
                "никогда не выдумывай и не оценивай значения. "
                "Если пользователь спрашивает о них — честно сообщи, "
                "что этих данных нет в загруженных источниках."
            )
        return "\n".join(lines)

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

        # --- Metric availability (Stage 4) ---
        lines.append("")
        lines.append(self.get_availability_section())

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
        unique = self._unique_activities
        d30 = self._today - timedelta(days=30)

        act_7d = [a for a in unique if d7 <= a.startTime.date() <= self._today]
        act_30d = [a for a in unique if d30 <= a.startTime.date() <= self._today]

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

        # --- Dedup note ---
        if self._activities_duplicates_removed > 0:
            lines.append(
                f"(Учтено после дедупликации: убрано "
                f"{self._activities_duplicates_removed} дубликатов из разных источников)"
            )

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
        wants_calories = self._detect_calories_intent(text)
        wants_rest_days = self._detect_rest_days_intent(text)
        wants_weekly = self._detect_weekly_load_intent(text)
        wants_progress = self._detect_progress_intent(text)
        wants_hrv = self._detect_hrv_intent(text)
        wants_sleep = self._detect_sleep_intent(text)
        wants_spo2 = self._detect_spo2_intent(text)
        wants_resting_hr = self._detect_resting_hr_intent(text)
        wants_skin_temp = self._detect_skin_temp_intent(text)
        wants_correlation = self._detect_correlation_intent(text)

        d_from, d_to = date_range if date_range else self._default_date_range()

        # --- Extended metrics with graceful degradation ---
        # These run before generic builders so an HRV query doesn't leak
        # through to the fallback.
        if wants_hrv:
            sections.append(
                self._build_extended_metric_analytics("hrvRmssdMilli", d_from, d_to)
            )
        if wants_sleep:
            sections.append(self._build_sleep_analytics(d_from, d_to))
        if wants_spo2:
            sections.append(
                self._build_extended_metric_analytics("spo2Percentage", d_from, d_to)
            )
        if wants_resting_hr:
            sections.append(
                self._build_extended_metric_analytics(
                    "restingHeartRate", d_from, d_to
                )
            )
        if wants_skin_temp:
            sections.append(
                self._build_extended_metric_analytics(
                    "skinTempCelsius", d_from, d_to
                )
            )

        # --- Build sections ---
        if wants_steps:
            sections.append(self._build_steps_analytics(d_from, d_to))

        if wants_calories:
            sections.append(self._build_calories_analytics(d_from, d_to))

        if wants_rest_days:
            sections.append(self._build_rest_days_analytics(d_from, d_to))

        if wants_weekly:
            sections.append(self._build_weekly_load_analytics(d_from, d_to))

        if wants_progress and (sport_types is None or "running" in sport_types):
            sections.append(self._build_running_progress_analytics(d_from, d_to))

        if wants_training:
            sections.append(self._build_training_list(d_from, d_to, sport_types))

        if wants_recovery:
            sections.append(self._build_recovery_analytics(d_from, d_to))

        if wants_correlation or (wants_recovery and wants_training):
            sections.append(self._build_correlations_analytics(d_from, d_to))

        if (
            sport_types
            and not wants_training
            and not wants_progress
            and not wants_weekly
        ):
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
                a for a in self._unique_activities
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
            [a for a in self._unique_activities
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

        unique = self._unique_activities
        recent_acts = [a for a in unique if d_from <= a.startTime.date() <= d_to]
        if recent_acts:
            by_sport = Counter(a.sportType for a in recent_acts)
            parts = [f"{s}: {c}" for s, c in sorted(by_sport.items())]
            lines.append(f"Тренировки: {len(recent_acts)} ({', '.join(parts)})")

        return "\n".join(lines)

    # ==================================================================
    # STAGE 3: CROSS-FILE ANALYTICS
    # ==================================================================

    def _build_calories_analytics(self, d_from: date, d_to: date) -> str:
        """Total calories burned in a period.

        Primary source: daily-facts.caloriesKcal (summed where present).
        Fallback for missing days: sum of deduplicated activities on that day.
        """
        lines = [f"--- КАЛОРИИ ({d_from} — {d_to}) ---"]

        facts_in_range = sorted(
            [f for f in self._daily_facts if d_from <= f.isoDate <= d_to],
            key=lambda f: f.isoDate,
        )
        if not facts_in_range and not any(
            d_from <= a.startTime.date() <= d_to for a in self._unique_activities
        ):
            lines.append("Нет данных о калориях за указанный период.")
            return "\n".join(lines)

        # Build per-day activity-calorie index from deduped activities
        act_by_day: dict[date, float] = defaultdict(float)
        for a in self._unique_activities:
            d = a.startTime.date()
            if d_from <= d <= d_to:
                act_by_day[d] += a.calories

        total_from_facts = 0.0
        total_from_activities = 0.0
        days_from_facts = 0
        days_from_activities = 0
        missing_days = 0

        per_day: list[tuple[date, float, str]] = []
        for f in facts_in_range:
            if f.caloriesKcal is not None:
                per_day.append((f.isoDate, f.caloriesKcal, "daily-facts"))
                total_from_facts += f.caloriesKcal
                days_from_facts += 1
            elif f.isoDate in act_by_day:
                val = act_by_day[f.isoDate]
                per_day.append((f.isoDate, val, "activities"))
                total_from_activities += val
                days_from_activities += 1
            else:
                per_day.append((f.isoDate, 0.0, "—"))
                missing_days += 1

        total = total_from_facts + total_from_activities
        total_days = days_from_facts + days_from_activities

        lines.append(f"Всего: {total:.0f} kcal за {total_days} дн.")
        if days_from_facts:
            lines.append(
                f"  из daily-facts: {total_from_facts:.0f} kcal "
                f"({days_from_facts} дн.)"
            )
        if days_from_activities:
            lines.append(
                f"  из активностей (fallback): {total_from_activities:.0f} kcal "
                f"({days_from_activities} дн.)"
            )
        if missing_days:
            lines.append(f"  нет данных: {missing_days} дн.")
        if total_days > 0:
            avg = total / total_days
            lines.append(f"В среднем: {avg:.0f} kcal/день")

        lines.append("")
        lines.append("По дням:")
        for d, val, src in per_day:
            if src == "—":
                lines.append(f"  {d}: — (нет данных)")
            else:
                lines.append(f"  {d}: {val:.0f} kcal [{src}]")

        return "\n".join(lines)

    def _build_rest_days_analytics(self, d_from: date, d_to: date) -> str:
        """Days in range with no training activity (after dedup)."""
        lines = [f"--- ДНИ ОТДЫХА ({d_from} — {d_to}) ---"]

        total_days = (d_to - d_from).days + 1
        if total_days <= 0:
            lines.append("Некорректный период.")
            return "\n".join(lines)

        active_days: set[date] = {
            a.startTime.date() for a in self._unique_activities
            if d_from <= a.startTime.date() <= d_to
        }

        # Only consider days that are covered by daily-facts (otherwise we
        # can't tell if there really was no training or just no data).
        covered_days: set[date] = {
            f.isoDate for f in self._daily_facts
            if d_from <= f.isoDate <= d_to
        }
        # If daily-facts don't cover the whole range, fall back to all days
        if not covered_days:
            covered_days = {
                d_from + timedelta(days=i) for i in range(total_days)
            }

        rest_days = sorted(covered_days - active_days)
        training_days = sorted(covered_days & active_days)

        lines.append(f"Период: {total_days} дн., данных по {len(covered_days)} дн.")
        lines.append(
            f"Тренировочных дней: {len(training_days)}, "
            f"дней отдыха: {len(rest_days)}"
        )

        if rest_days:
            lines.append("")
            lines.append("Дни отдыха:")
            for d in rest_days:
                lines.append(f"  {d}")

        return "\n".join(lines)

    def _build_weekly_load_analytics(self, d_from: date, d_to: date) -> str:
        """Weekly training load: count, total duration, total calories per week."""
        lines = [f"--- НЕДЕЛЬНАЯ НАГРУЗКА ({d_from} — {d_to}) ---"]

        acts = [
            a for a in self._unique_activities
            if d_from <= a.startTime.date() <= d_to
        ]
        if not acts:
            lines.append("Нет тренировок за указанный период.")
            return "\n".join(lines)

        # Group by ISO week (Monday of the week)
        by_week: dict[date, list[Activity]] = defaultdict(list)
        for a in acts:
            d = a.startTime.date()
            monday = d - timedelta(days=d.weekday())
            by_week[monday].append(a)

        lines.append(f"Всего тренировок (после дедупликации): {len(acts)}")
        lines.append("")
        lines.append("По неделям (начало недели — понедельник):")
        for monday in sorted(by_week.keys()):
            week_acts = by_week[monday]
            sunday = monday + timedelta(days=6)
            total_dur = sum(a.duration for a in week_acts)
            total_cal = sum(a.calories for a in week_acts)
            total_dist = sum(a.distance for a in week_acts)
            by_sport = Counter(a.sportType for a in week_acts)
            sport_str = ", ".join(
                f"{s}: {c}" for s, c in sorted(by_sport.items())
            )
            lines.append(
                f"  {monday} — {sunday}: {len(week_acts)} трен. "
                f"({sport_str}), {_fmt_duration(total_dur)}, "
                f"{_fmt_distance(total_dist)}, {total_cal:.0f} kcal"
            )

        return "\n".join(lines)

    def _build_running_progress_analytics(
        self, d_from: date, d_to: date,
    ) -> str:
        """Running progress: compare average pace across periods.

        Pace is computed as seconds per kilometre, averaged over activities
        with non-zero distance. Also splits the range in two halves to show
        short-term trend.
        """
        lines = [f"--- ПРОГРЕСС В БЕГЕ ({d_from} — {d_to}) ---"]

        runs = [
            a for a in self._unique_activities
            if a.sportType == "running"
            and d_from <= a.startTime.date() <= d_to
            and a.distance > 0
            and a.duration > 0
        ]

        skipped_no_dist = sum(
            1 for a in self._unique_activities
            if a.sportType == "running"
            and d_from <= a.startTime.date() <= d_to
            and (a.distance <= 0 or a.duration <= 0)
        )

        if not runs:
            lines.append("Нет пробежек с дистанцией за указанный период.")
            if skipped_no_dist:
                lines.append(
                    f"(Пропущено {skipped_no_dist} без дистанции/длительности)"
                )
            return "\n".join(lines)

        runs.sort(key=lambda a: a.startTime)

        def pace_sec_per_km(a: Activity) -> float:
            return a.duration / (a.distance / 1000)

        def _fmt_pace(sec_per_km: float) -> str:
            m = int(sec_per_km // 60)
            s = int(sec_per_km % 60)
            return f"{m}:{s:02d} мин/км"

        total_distance = sum(a.distance for a in runs)
        total_duration = sum(a.duration for a in runs)
        overall_pace = total_duration / (total_distance / 1000)

        lines.append(f"Пробежек: {len(runs)}")
        lines.append(f"Общая дистанция: {_fmt_distance(total_distance)}")
        lines.append(f"Общее время: {_fmt_duration(total_duration)}")
        lines.append(f"Средний темп: {_fmt_pace(overall_pace)}")

        # Compare first half vs second half (by count)
        if len(runs) >= 2:
            half = len(runs) // 2
            first = runs[:half] if half > 0 else runs[:1]
            second = runs[half:] if half > 0 else runs[1:]
            if first and second:
                p1 = sum(a.duration for a in first) / (
                    sum(a.distance for a in first) / 1000
                )
                p2 = sum(a.duration for a in second) / (
                    sum(a.distance for a in second) / 1000
                )
                lines.append("")
                lines.append(
                    f"Первая половина ({len(first)} пробежек): {_fmt_pace(p1)}"
                )
                lines.append(
                    f"Вторая половина ({len(second)} пробежек): {_fmt_pace(p2)}"
                )
                diff = p2 - p1
                if diff < -5:
                    lines.append("Темп улучшается: ↑ быстрее")
                elif diff > 5:
                    lines.append("Темп замедляется: ↓ медленнее")
                else:
                    lines.append("Темп стабилен: →")

        lines.append("")
        lines.append("По пробежкам:")
        for a in runs:
            d_str = a.startTime.strftime("%Y-%m-%d")
            lines.append(
                f"  {d_str} | {_fmt_distance(a.distance)} | "
                f"{_fmt_duration(a.duration)} | {_fmt_pace(pace_sec_per_km(a))}"
            )

        if skipped_no_dist:
            lines.append("")
            lines.append(
                f"(Пропущено {skipped_no_dist} пробежек без дистанции)"
            )

        return "\n".join(lines)

    # ==================================================================
    # STAGE 4: EXTENDED METRICS + CORRELATIONS (graceful degradation)
    # ==================================================================

    def _build_extended_metric_analytics(
        self, metric: str, d_from: date, d_to: date,
    ) -> str:
        """Generic builder for single-value extended metrics.

        Used for HRV, SpO2, resting HR, skin temperature. If the metric
        has no data anywhere in the dataset, returns an honest
        "нет данных" response without inventing values.
        """
        label = _METRIC_LABELS_RU.get(metric, metric)
        header = f"--- {label.upper()} ({d_from} — {d_to}) ---"

        if not self.is_metric_available(metric):
            return (
                f"{header}\n"
                f"Данные по метрике «{label}» отсутствуют в загруженных "
                f"источниках. Невозможно рассчитать или оценить значения."
            )

        facts = sorted(
            [
                f for f in self._daily_facts
                if d_from <= f.isoDate <= d_to
                and getattr(f, metric, None) is not None
            ],
            key=lambda f: f.isoDate,
        )

        lines = [header]
        if not facts:
            lines.append(
                f"Нет данных по «{label}» за указанный период "
                "(хотя метрика присутствует в других периодах)."
            )
            return "\n".join(lines)

        values = [float(getattr(f, metric)) for f in facts]
        avg = sum(values) / len(values)
        lines.append(f"Записей: {len(facts)}")
        lines.append(f"Среднее: {avg:.1f}")
        lines.append(f"Минимум: {min(values):.1f}")
        lines.append(f"Максимум: {max(values):.1f}")
        if len(values) >= 2:
            lines.append(f"Тренд: {_compute_trend(values)}")

        lines.append("")
        lines.append("По дням:")
        for f, v in zip(facts, values):
            lines.append(f"  {f.isoDate}: {v:.1f}")

        return "\n".join(lines)

    def _build_sleep_analytics(self, d_from: date, d_to: date) -> str:
        """Sleep analytics combining performance % and total time in bed.

        Reports both metrics if available, falls back gracefully when
        either or both are missing.
        """
        header = f"--- СОН ({d_from} — {d_to}) ---"
        perf_available = self.is_metric_available("sleepPerformancePercentage")
        dur_available = self.is_metric_available("sleepTotalInBedTimeMilli")

        if not perf_available and not dur_available:
            return (
                f"{header}\n"
                "Данные о сне (эффективность и длительность) отсутствуют "
                "в загруженных источниках. Невозможно рассчитать или "
                "оценить значения."
            )

        facts = sorted(
            [f for f in self._daily_facts if d_from <= f.isoDate <= d_to],
            key=lambda f: f.isoDate,
        )

        lines = [header]

        if perf_available:
            perf_facts = [
                f for f in facts if f.sleepPerformancePercentage is not None
            ]
            if perf_facts:
                perfs = [f.sleepPerformancePercentage for f in perf_facts]
                avg = sum(perfs) / len(perfs)
                lines.append(
                    f"Эффективность сна: среднее {avg:.0f}%, "
                    f"мин {min(perfs):.0f}%, макс {max(perfs):.0f}% "
                    f"({len(perf_facts)} дн.)"
                )
            else:
                lines.append(
                    "Эффективность сна: нет данных за указанный период"
                )
        else:
            lines.append(
                "Эффективность сна: данных в источниках нет"
            )

        if dur_available:
            dur_facts = [
                f for f in facts if f.sleepTotalInBedTimeMilli is not None
            ]
            if dur_facts:
                total_ms = sum(f.sleepTotalInBedTimeMilli for f in dur_facts)
                avg_ms = total_ms / len(dur_facts)
                avg_hours = avg_ms / 1000 / 3600
                lines.append(
                    f"Длительность сна: в среднем {avg_hours:.1f} ч "
                    f"({len(dur_facts)} дн.)"
                )
            else:
                lines.append(
                    "Длительность сна: нет данных за указанный период"
                )
        else:
            lines.append(
                "Длительность сна: данных в источниках нет"
            )

        return "\n".join(lines)

    def _build_correlations_analytics(
        self, d_from: date, d_to: date,
    ) -> str:
        """Correlations between recovery / sleep / HRV and training load.

        Runs only those sub-sections for which data is actually present,
        so it degrades gracefully when extended metrics are missing.
        """
        lines = [f"--- КОРРЕЛЯЦИИ ({d_from} — {d_to}) ---"]

        # Build per-day training load from unique activities
        load_by_day: dict[date, float] = defaultdict(float)
        duration_by_day: dict[date, float] = defaultdict(float)
        for a in self._unique_activities:
            d = a.startTime.date()
            if d_from <= d <= d_to:
                load_by_day[d] += a.calories
                duration_by_day[d] += a.duration

        def _series(metric: str) -> tuple[list[float], list[float]]:
            xs: list[float] = []
            ys: list[float] = []
            for f in self._daily_facts:
                if not (d_from <= f.isoDate <= d_to):
                    continue
                v = getattr(f, metric, None)
                if v is None:
                    continue
                # Align with training load on the same day (can be 0)
                xs.append(float(v))
                ys.append(float(load_by_day.get(f.isoDate, 0.0)))
            return xs, ys

        sub_lines: list[str] = []

        # 1. Recovery vs training load (calories on same day)
        if self.is_metric_available("recoveryScore"):
            xs, ys = _series("recoveryScore")
            if len(xs) >= 3:
                r = _pearson(xs, ys)
                sub_lines.append(
                    f"Восстановление ↔ нагрузка (kcal): "
                    f"r = {r:.2f}, n = {len(xs)} "
                    f"({_interpret_correlation(r)})"
                )
            else:
                sub_lines.append(
                    "Восстановление ↔ нагрузка: недостаточно парных "
                    "наблюдений (нужно >= 3)"
                )
        else:
            sub_lines.append(
                "Восстановление ↔ нагрузка: данных о восстановлении нет"
            )

        # 2. Sleep vs recovery (both optional)
        if (
            self.is_metric_available("sleepPerformancePercentage")
            and self.is_metric_available("recoveryScore")
        ):
            xs: list[float] = []
            ys: list[float] = []
            for f in self._daily_facts:
                if not (d_from <= f.isoDate <= d_to):
                    continue
                if (
                    f.sleepPerformancePercentage is not None
                    and f.recoveryScore is not None
                ):
                    xs.append(f.sleepPerformancePercentage)
                    ys.append(f.recoveryScore)
            if len(xs) >= 3:
                r = _pearson(xs, ys)
                sub_lines.append(
                    f"Сон ↔ восстановление: r = {r:.2f}, n = {len(xs)} "
                    f"({_interpret_correlation(r)})"
                )
            else:
                sub_lines.append(
                    "Сон ↔ восстановление: недостаточно парных наблюдений"
                )
        else:
            sub_lines.append(
                "Сон ↔ восстановление: одной из метрик нет в источниках"
            )

        # 3. HRV vs training intensity (duration)
        if self.is_metric_available("hrvRmssdMilli"):
            xs: list[float] = []
            ys: list[float] = []
            for f in self._daily_facts:
                if not (d_from <= f.isoDate <= d_to):
                    continue
                if f.hrvRmssdMilli is not None:
                    xs.append(float(f.hrvRmssdMilli))
                    ys.append(float(duration_by_day.get(f.isoDate, 0.0)))
            if len(xs) >= 3:
                r = _pearson(xs, ys)
                sub_lines.append(
                    f"HRV ↔ длительность тренировок: r = {r:.2f}, "
                    f"n = {len(xs)} ({_interpret_correlation(r)})"
                )
            else:
                sub_lines.append(
                    "HRV ↔ длительность тренировок: недостаточно "
                    "парных наблюдений"
                )
        else:
            sub_lines.append(
                "HRV ↔ длительность тренировок: данных HRV нет в источниках"
            )

        lines.extend(sub_lines)
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

    def _detect_calories_intent(self, text: str) -> bool:
        return any(kw in text for kw in ["калори", "ккал", "kcal", "calories"])

    def _detect_rest_days_intent(self, text: str) -> bool:
        keywords = ["день отдыха", "дни отдыха", "дней отдыха",
                    "отдых", "rest day", "без трениров"]
        return any(kw in text for kw in keywords)

    def _detect_weekly_load_intent(self, text: str) -> bool:
        keywords = [
            "нагрузк", "объём трениров", "объем трениров",
            "недельн", "по недел", "weekly", "training load",
        ]
        return any(kw in text for kw in keywords)

    def _detect_progress_intent(self, text: str) -> bool:
        keywords = [
            "прогресс", "темп", "pace", "улучш", "progress",
            "быстрее", "медленнее",
        ]
        return any(kw in text for kw in keywords)

    def _detect_hrv_intent(self, text: str) -> bool:
        keywords = ["hrv", "вариабельност", "rmssd"]
        return any(kw in text for kw in keywords)

    def _detect_sleep_intent(self, text: str) -> bool:
        # "спал", "сон", "сна", "sleep"
        keywords = ["сон", "сна", "спал", "сну", "сном", "sleep"]
        return any(kw in text for kw in keywords)

    def _detect_spo2_intent(self, text: str) -> bool:
        keywords = ["spo2", "сатурац", "кислород", "насыщени"]
        return any(kw in text for kw in keywords)

    def _detect_resting_hr_intent(self, text: str) -> bool:
        keywords = [
            "пульс в покое", "пульса в покое", "rhr",
            "resting heart", "resting hr", "покойный пульс",
        ]
        return any(kw in text for kw in keywords)

    def _detect_skin_temp_intent(self, text: str) -> bool:
        keywords = [
            "температура кожи", "температуру кожи", "температуры кожи",
            "skin temp", "skin temperature",
        ]
        return any(kw in text for kw in keywords)

    def _detect_correlation_intent(self, text: str) -> bool:
        keywords = [
            "корреляц", "связь", "влиян", "зависимост",
            "correlation", "correlate",
        ]
        return any(kw in text for kw in keywords)

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


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient for two equal-length series.

    Returns 0.0 if inputs are degenerate (length < 2 or zero variance).
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0
    return num / denom


def _interpret_correlation(r: float) -> str:
    """Human-readable verdict for a Pearson r value."""
    if r >= 0.7:
        return "сильная положительная"
    if r >= 0.4:
        return "умеренная положительная"
    if r >= 0.15:
        return "слабая положительная"
    if r <= -0.7:
        return "сильная отрицательная"
    if r <= -0.4:
        return "умеренная отрицательная"
    if r <= -0.15:
        return "слабая отрицательная"
    return "связи не обнаружено"


# Module-level singleton
sports_service = SportsService()
