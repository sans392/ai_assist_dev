"""Pydantic v2 models for health/sports data validation.

Three data sources:
- activities.json   — individual workouts (WHOOP, Apple Health, manual)
- daily-facts.json  — daily aggregated metrics (steps, calories, recovery)
- health-days.json  — sync metadata per platform per day
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

logger = logging.getLogger(__name__)


class ActivityFlag(str, Enum):
    """Flags for anomalous or suspicious activity records."""
    TEST_DATA = "test_data"
    SHORT_DURATION = "short_duration"
    INVALID_TIME_RANGE = "invalid_time_range"
    NEGATIVE_CALORIES = "negative_calories"


class Activity(BaseModel):
    """Single workout / activity record."""
    id: str
    title: str = ""
    sportType: str = "activity"
    distance: float = 0
    duration: int = 0  # seconds
    startTime: datetime
    endTime: datetime
    avgSpeed: float = 0
    maxSpeed: float = 0
    elevation: float = 0
    calories: float = 0

    # Populated after validation
    flags: list[ActivityFlag] = []

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def detect_anomalies(self) -> Activity:
        flags: list[ActivityFlag] = []
        title_lower = self.title.lower()
        if any(kw in title_lower for kw in ("debug", "fake", "test")):
            flags.append(ActivityFlag.TEST_DATA)
        if self.duration < 30:
            flags.append(ActivityFlag.SHORT_DURATION)
        if self.endTime < self.startTime:
            flags.append(ActivityFlag.INVALID_TIME_RANGE)
        if self.calories < 0:
            flags.append(ActivityFlag.NEGATIVE_CALORIES)
        self.flags = flags
        return self


class DailyFact(BaseModel):
    """Daily aggregated health metrics."""
    id: str
    isoDate: date
    steps: int = 0
    caloriesKcal: float | None = None
    recoveryScore: float | None = None
    sleepPerformancePercentage: float | None = None
    sourcesJson: str = "{}"

    # Parsed version of sourcesJson
    sources: dict[str, str] = {}

    # Flags
    flags: list[str] = []

    model_config = ConfigDict(extra="ignore")

    @field_validator("sourcesJson", mode="before")
    @classmethod
    def ensure_sources_string(cls, v: object) -> str:
        if isinstance(v, dict):
            return json.dumps(v)
        return str(v) if v is not None else "{}"

    @model_validator(mode="after")
    def parse_and_validate(self) -> DailyFact:
        # Parse sourcesJson into dict
        try:
            self.sources = json.loads(self.sourcesJson)
        except (json.JSONDecodeError, TypeError):
            self.sources = {}
            self.flags.append("invalid_sources_json")

        # Validate recoveryScore range
        if self.recoveryScore is not None:
            if not (0 <= self.recoveryScore <= 100):
                self.flags.append("recovery_out_of_range")

        return self


class HealthDay(BaseModel):
    """Sync/fetch metadata for a platform on a given day."""
    id: str
    isoDate: date
    platformType: str
    fetchedAt: datetime
    deletedAt: datetime | None = None

    model_config = ConfigDict(extra="ignore")


class DataLoadReport(BaseModel):
    """Summary of data loading results."""
    activities_loaded: int = 0
    activities_skipped: int = 0
    activities_flagged: int = 0
    activities_unique: int = 0
    activities_duplicates_removed: int = 0
    daily_facts_loaded: int = 0
    daily_facts_skipped: int = 0
    health_days_loaded: int = 0
    health_days_skipped: int = 0
    date_range_start: date | None = None
    date_range_end: date | None = None
    available_metrics: list[str] = []
    sport_types: list[str] = []
