"""Service for loading, validating, and querying health/sports data.

Loads data from three JSON files (activities, daily-facts, health-days),
validates via Pydantic models, flags anomalies, and provides query methods.

Stage 1: data loading + validation only. Analytics come in Stage 2+.
"""

from __future__ import annotations

import json
import logging
from datetime import date
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
    # Public: summary and analytics (Stage 2+)
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Compact summary for system prompt. Minimal in Stage 1."""
        if not self._daily_facts and not self._activities:
            return "Данные о здоровье и тренировках отсутствуют."

        report = self._load_report
        lines = ["--- СВОДКА ПО ДАННЫМ ---"]
        lines.append(f"Период данных: {report.date_range_start} — {report.date_range_end}")
        lines.append(f"Тренировок загружено: {report.activities_loaded}")
        lines.append(f"Виды спорта: {', '.join(report.sport_types)}")
        lines.append(f"Дней с метриками: {report.daily_facts_loaded}")
        return "\n".join(lines)

    def get_analytics(self, user_message: str) -> str:
        """Context-aware analytics. Stub for Stage 1."""
        return ""


# Module-level singleton
sports_service = SportsService()
