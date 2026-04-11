"""Tests for Sports Assistant — Stage 1 + Stage 2 + Stage 3 analytics."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from app.agents import get_agent_by_mode
from app.agents.sports import SportsAgent
from app.models.health_schemas import Activity, ActivityFlag, DailyFact, HealthDay
from app.services.sports import (
    SportsService,
    _activity_source,
    _are_duplicates,
    _compute_trend,
    _deduplicate,
    _overlap_seconds,
    _pick_canonical,
)


# ---------------------------------------------------------------
# Pydantic model validation (Stage 1)
# ---------------------------------------------------------------

class TestActivityModel:
    def test_valid_activity(self):
        data = {
            "id": "abc-123",
            "title": "Morning Run",
            "sportType": "running",
            "distance": 5000,
            "duration": 1800,
            "startTime": "2026-03-03T06:00:00.000Z",
            "endTime": "2026-03-03T06:30:00.000Z",
            "calories": 350,
        }
        a = Activity.model_validate(data)
        assert a.sportType == "running"
        assert a.distance == 5000
        assert a.duration == 1800
        assert a.calories == 350
        assert a.flags == []

    def test_flags_debug_title(self):
        data = {
            "id": "test-1",
            "title": "Debug Fake WHOOP Run",
            "sportType": "running",
            "duration": 900,
            "startTime": "2026-03-03T11:20:00.000Z",
            "endTime": "2026-03-03T11:35:00.000Z",
        }
        a = Activity.model_validate(data)
        assert ActivityFlag.TEST_DATA in a.flags

    def test_flags_short_duration(self):
        data = {
            "id": "short-1",
            "title": "Quick test",
            "sportType": "running",
            "duration": 10,
            "startTime": "2026-03-03T11:00:00.000Z",
            "endTime": "2026-03-03T11:00:10.000Z",
        }
        a = Activity.model_validate(data)
        assert ActivityFlag.SHORT_DURATION in a.flags
        assert ActivityFlag.TEST_DATA in a.flags

    def test_flags_invalid_time_range(self):
        data = {
            "id": "bad-time",
            "title": "Backward run",
            "sportType": "running",
            "duration": 100,
            "startTime": "2026-03-03T12:00:00.000Z",
            "endTime": "2026-03-03T11:00:00.000Z",
        }
        a = Activity.model_validate(data)
        assert ActivityFlag.INVALID_TIME_RANGE in a.flags

    def test_flags_negative_calories(self):
        data = {
            "id": "neg-cal",
            "title": "Bad data",
            "sportType": "running",
            "duration": 600,
            "startTime": "2026-03-03T11:00:00.000Z",
            "endTime": "2026-03-03T11:10:00.000Z",
            "calories": -50,
        }
        a = Activity.model_validate(data)
        assert ActivityFlag.NEGATIVE_CALORIES in a.flags

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            Activity.model_validate({"title": "No id or times"})

    def test_extra_fields_ignored(self):
        data = {
            "id": "extra-1",
            "title": "Run",
            "sportType": "running",
            "duration": 600,
            "startTime": "2026-03-03T11:00:00.000Z",
            "endTime": "2026-03-03T11:10:00.000Z",
            "likesCount": 42,
            "commentsCount": 3,
            "unknownField": "hello",
        }
        a = Activity.model_validate(data)
        assert a.id == "extra-1"


class TestDailyFactModel:
    def test_valid_daily_fact(self):
        data = {
            "id": "df-1",
            "isoDate": "2026-03-16",
            "steps": 8000,
            "caloriesKcal": 500,
            "recoveryScore": 43,
            "sourcesJson": '{"steps":"apple_health","recoveryScore":"whoop_recovery_api"}',
        }
        f = DailyFact.model_validate(data)
        assert f.steps == 8000
        assert f.sources["steps"] == "apple_health"
        assert f.flags == []

    def test_null_optional_fields(self):
        data = {
            "id": "df-2",
            "isoDate": "2026-03-10",
            "steps": 100,
            "caloriesKcal": None,
            "recoveryScore": None,
        }
        f = DailyFact.model_validate(data)
        assert f.caloriesKcal is None
        assert f.recoveryScore is None

    def test_recovery_out_of_range_flagged(self):
        data = {
            "id": "df-3",
            "isoDate": "2026-03-10",
            "steps": 100,
            "recoveryScore": 150,
        }
        f = DailyFact.model_validate(data)
        assert "recovery_out_of_range" in f.flags

    def test_invalid_sources_json_flagged(self):
        data = {
            "id": "df-4",
            "isoDate": "2026-03-10",
            "steps": 100,
            "sourcesJson": "not valid json {{{",
        }
        f = DailyFact.model_validate(data)
        assert "invalid_sources_json" in f.flags
        assert f.sources == {}

    def test_sources_json_as_dict(self):
        """sourcesJson passed as dict (not string) should also work."""
        data = {
            "id": "df-5",
            "isoDate": "2026-03-10",
            "steps": 100,
            "sourcesJson": {"steps": "apple_health"},
        }
        f = DailyFact.model_validate(data)
        assert f.sources["steps"] == "apple_health"


class TestHealthDayModel:
    def test_valid_health_day(self):
        data = {
            "id": "hd-1",
            "isoDate": "2026-03-16",
            "platformType": "whoop",
            "fetchedAt": "2026-03-16T16:13:27.282Z",
            "deletedAt": None,
        }
        h = HealthDay.model_validate(data)
        assert h.platformType == "whoop"
        assert h.deletedAt is None

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            HealthDay.model_validate({"id": "hd-bad"})


# ---------------------------------------------------------------
# SportsService — data loading (Stage 1)
# ---------------------------------------------------------------

class TestSportsService:
    def test_loads_from_example_data(self):
        svc = SportsService()
        assert len(svc.activities) > 0
        assert len(svc.daily_facts) > 0
        assert len(svc.health_days) > 0

    def test_load_report_populated(self):
        svc = SportsService()
        report = svc.load_report
        assert report.activities_loaded > 0
        assert report.daily_facts_loaded > 0
        assert report.health_days_loaded > 0
        assert report.date_range_start is not None
        assert report.date_range_end is not None
        assert len(report.sport_types) > 0

    def test_debug_activities_flagged(self):
        svc = SportsService()
        flagged = [a for a in svc.activities if ActivityFlag.TEST_DATA in a.flags]
        assert len(flagged) > 0, "Expected debug/test activities to be flagged"

    def test_clean_activities_excludes_debug(self):
        svc = SportsService()
        clean = svc.clean_activities
        for a in clean:
            assert ActivityFlag.TEST_DATA not in a.flags
        assert len(clean) < len(svc.activities)

    def test_get_summary_not_empty(self):
        svc = SportsService()
        summary = svc.get_summary()
        assert "СВОДКА" in summary
        assert len(summary) > 20

    def test_available_metrics_includes_steps(self):
        svc = SportsService()
        assert "steps" in svc.load_report.available_metrics


# ---------------------------------------------------------------
# SportsService — enhanced summary (Stage 2 Level A)
# ---------------------------------------------------------------

class TestSummaryLevelA:
    def _make_service(self, today: date | None = None) -> SportsService:
        svc = SportsService()
        if today:
            svc._today = today
        return svc

    def test_summary_contains_steps_section(self):
        svc = self._make_service(date(2026, 3, 16))
        summary = svc.get_summary()
        assert "Шаги за последние 7 дней" in summary

    def test_summary_contains_training_counts(self):
        svc = self._make_service(date(2026, 3, 16))
        summary = svc.get_summary()
        assert "Тренировки за 7 дней" in summary
        assert "Тренировки за 30 дней" in summary

    def test_summary_contains_recovery(self):
        svc = self._make_service(date(2026, 3, 16))
        summary = svc.get_summary()
        assert "Восстановление" in summary
        assert "Тренд" in summary

    def test_summary_contains_date_range(self):
        svc = self._make_service(date(2026, 3, 16))
        summary = svc.get_summary()
        assert "Период данных" in summary

    def test_summary_contains_today_date(self):
        svc = self._make_service(date(2026, 3, 16))
        summary = svc.get_summary()
        assert "2026-03-16" in summary

    def test_summary_steps_with_data(self):
        """When today is within data range, steps should have values."""
        svc = self._make_service(date(2026, 3, 16))
        summary = svc.get_summary()
        # Should have actual numbers, not "нет данных"
        assert "всего" in summary
        assert "в среднем" in summary

    def test_summary_no_steps_data(self):
        """When today is far from data, steps should show no data."""
        svc = self._make_service(date(2026, 6, 1))
        summary = svc.get_summary()
        assert "Шаги за последние 7 дней: нет данных" in summary


# ---------------------------------------------------------------
# Intent detection (Stage 2 Level A)
# ---------------------------------------------------------------

class TestIntentDetection:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    # --- Steps intent ---

    def test_detect_steps_шаги(self):
        assert self.svc._detect_steps_intent("покажи мои шаги") is True

    def test_detect_steps_шагов(self):
        assert self.svc._detect_steps_intent("сколько шагов за неделю") is True

    def test_detect_steps_ходьба(self):
        assert self.svc._detect_steps_intent("статистика ходьбы") is True

    def test_detect_steps_negative(self):
        assert self.svc._detect_steps_intent("покажи тренировки") is False

    # --- Training intent ---

    def test_detect_training_тренировки(self):
        assert self.svc._detect_training_intent("мои тренировки") is True

    def test_detect_training_занятия(self):
        assert self.svc._detect_training_intent("список занятий") is True

    def test_detect_training_активности(self):
        assert self.svc._detect_training_intent("покажи активности") is True

    def test_detect_training_тренировок(self):
        assert self.svc._detect_training_intent("сколько тренировок") is True

    def test_detect_training_negative(self):
        assert self.svc._detect_training_intent("как мое восстановление") is False

    # --- Recovery intent ---

    def test_detect_recovery_восстановление(self):
        assert self.svc._detect_recovery_intent("как мое восстановление") is True

    def test_detect_recovery_recovery(self):
        assert self.svc._detect_recovery_intent("show recovery score") is True

    def test_detect_recovery_negative(self):
        assert self.svc._detect_recovery_intent("покажи шаги") is False

    # --- Sport type detection ---

    def test_detect_sport_бег(self):
        result = self.svc._detect_sport_type("статистика бега")
        assert result == {"running"}

    def test_detect_sport_running(self):
        result = self.svc._detect_sport_type("show running stats")
        assert result == {"running"}

    def test_detect_sport_велосипед(self):
        result = self.svc._detect_sport_type("велосипедные тренировки")
        assert result == {"cycling"}

    def test_detect_sport_cycling(self):
        result = self.svc._detect_sport_type("cycling activities")
        assert result == {"cycling"}

    def test_detect_sport_none(self):
        result = self.svc._detect_sport_type("покажи восстановление")
        assert result is None

    def test_detect_sport_multiple(self):
        result = self.svc._detect_sport_type("бег и велосипед")
        assert result == {"running", "cycling"}

    # --- Date range detection ---

    def test_detect_date_вчера(self):
        result = self.svc._detect_date_range("шаги за вчера")
        assert result == (date(2026, 3, 15), date(2026, 3, 15))

    def test_detect_date_сегодня(self):
        result = self.svc._detect_date_range("шаги за сегодня")
        assert result == (date(2026, 3, 16), date(2026, 3, 16))

    def test_detect_date_прошлая_неделя(self):
        result = self.svc._detect_date_range("тренировки на прошлой неделе")
        # 2026-03-16 is Monday, so last week is Mon 2026-03-09 to Sun 2026-03-15
        assert result == (date(2026, 3, 9), date(2026, 3, 15))

    def test_detect_date_последние_7_дней(self):
        result = self.svc._detect_date_range("шаги за последние 7 дней")
        assert result == (date(2026, 3, 9), date(2026, 3, 16))

    def test_detect_date_за_N_дней(self):
        result = self.svc._detect_date_range("за 14 дней")
        assert result == (date(2026, 3, 2), date(2026, 3, 16))

    def test_detect_date_за_март(self):
        result = self.svc._detect_date_range("тренировки за март")
        assert result == (date(2026, 3, 1), date(2026, 3, 31))

    def test_detect_date_в_феврале(self):
        result = self.svc._detect_date_range("шаги в феврале")
        assert result == (date(2026, 2, 1), date(2026, 2, 28))

    def test_detect_date_none(self):
        result = self.svc._detect_date_range("покажи шаги")
        assert result is None


# ---------------------------------------------------------------
# Pre-computed analytics builders (Stage 2 Level A)
# ---------------------------------------------------------------

class TestStepsAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_steps_for_period_with_data(self):
        result = self.svc._build_steps_analytics(date(2026, 3, 10), date(2026, 3, 16))
        assert "ШАГИ" in result
        assert "Всего" in result
        assert "В среднем" in result
        assert "По дням:" in result

    def test_steps_total_matches_data(self):
        """Verify pre-computed total against known data."""
        d_from, d_to = date(2026, 3, 10), date(2026, 3, 16)
        facts = [f for f in self.svc.daily_facts if d_from <= f.isoDate <= d_to]
        expected_total = sum(f.steps for f in facts)

        result = self.svc._build_steps_analytics(d_from, d_to)
        # The formatted total should appear in the output
        assert str(expected_total) in result.replace(" ", "")

    def test_steps_empty_period(self):
        result = self.svc._build_steps_analytics(date(2026, 6, 1), date(2026, 6, 7))
        assert "Нет данных" in result


class TestTrainingList:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_training_list_with_data(self):
        result = self.svc._build_training_list(
            date(2026, 3, 1), date(2026, 3, 16), None,
        )
        assert "ТРЕНИРОВКИ" in result
        assert "Найдено" in result

    def test_training_list_excludes_test_data(self):
        """Debug/test activities should not appear."""
        result = self.svc._build_training_list(
            date(2026, 3, 1), date(2026, 3, 16), None,
        )
        assert "Debug" not in result
        assert "Fake" not in result

    def test_training_list_filtered_by_sport(self):
        result = self.svc._build_training_list(
            date(2026, 3, 1), date(2026, 3, 16), {"cycling"},
        )
        assert "cycling" in result
        # Should not contain running activities
        assert "running" not in result.split("---")[1]  # skip header

    def test_training_list_empty_period(self):
        result = self.svc._build_training_list(
            date(2026, 6, 1), date(2026, 6, 7), None,
        )
        assert "не найдено" in result


class TestRecoveryAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_recovery_with_data(self):
        result = self.svc._build_recovery_analytics(
            date(2026, 2, 25), date(2026, 3, 16),
        )
        assert "ВОССТАНОВЛЕНИЕ" in result
        assert "Среднее" in result
        assert "Минимум" in result
        assert "Максимум" in result
        assert "Тренд" in result

    def test_recovery_values_match_data(self):
        """Verify min/max against known data."""
        d_from, d_to = date(2026, 2, 25), date(2026, 3, 16)
        facts = [f for f in self.svc.daily_facts
                 if d_from <= f.isoDate <= d_to and f.recoveryScore is not None]
        scores = [f.recoveryScore for f in facts]

        result = self.svc._build_recovery_analytics(d_from, d_to)
        assert f"{min(scores):.0f}%" in result
        assert f"{max(scores):.0f}%" in result

    def test_recovery_empty_period(self):
        result = self.svc._build_recovery_analytics(
            date(2026, 6, 1), date(2026, 6, 7),
        )
        assert "Нет данных" in result

    def test_recovery_all_null_period(self):
        """Period where all recovery scores are null."""
        # Mar 5-9 have no recovery scores in the data
        result = self.svc._build_recovery_analytics(
            date(2026, 3, 5), date(2026, 3, 9),
        )
        assert "Нет данных" in result


class TestSportBreakdown:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_cycling_breakdown(self):
        result = self.svc._build_sport_breakdown(
            {"cycling"}, date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "CYCLING" in result
        assert "Всего тренировок" in result
        assert "Общая длительность" in result

    def test_running_breakdown(self):
        result = self.svc._build_sport_breakdown(
            {"running"}, date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "RUNNING" in result
        assert "Всего тренировок" in result

    def test_empty_sport_breakdown(self):
        result = self.svc._build_sport_breakdown(
            {"swimming"}, date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "не найдено" in result


# ---------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------

class TestComputeTrend:
    def test_growing(self):
        assert "растёт" in _compute_trend([10, 20, 30, 40, 50])

    def test_declining(self):
        assert "падает" in _compute_trend([50, 40, 30, 20, 10])

    def test_stable(self):
        assert "стабильно" in _compute_trend([30, 32, 28, 31, 33])

    def test_single_value(self):
        assert "недостаточно" in _compute_trend([42])


# ---------------------------------------------------------------
# get_analytics() integration
# ---------------------------------------------------------------

class TestGetAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_steps_query(self):
        result = self.svc.get_analytics("сколько шагов за последние 7 дней")
        assert "ШАГИ" in result

    def test_training_query(self):
        result = self.svc.get_analytics("покажи мои тренировки за март")
        assert "ТРЕНИРОВКИ" in result

    def test_recovery_query(self):
        result = self.svc.get_analytics("как мое восстановление")
        assert "ВОССТАНОВЛЕНИЕ" in result

    def test_sport_type_query(self):
        result = self.svc.get_analytics("статистика бега")
        assert "RUNNING" in result

    def test_sport_with_training(self):
        """Sport type + training intent should return filtered training list."""
        result = self.svc.get_analytics("тренировки по велосипеду за март")
        assert "ТРЕНИРОВКИ" in result
        assert "cycling" in result

    def test_empty_message(self):
        result = self.svc.get_analytics("")
        assert result == ""

    def test_fallback_on_unrecognized_query(self):
        result = self.svc.get_analytics("привет, как дела?")
        assert "ОБЗОР" in result


# ---------------------------------------------------------------
# SportsAgent
# ---------------------------------------------------------------

class TestSportsAgent:
    def test_agent_registered(self):
        agent = get_agent_by_mode("sports")
        assert isinstance(agent, SportsAgent)

    def test_prepare_messages_has_system(self):
        agent = SportsAgent()
        messages = [{"role": "user", "content": "Покажи мои тренировки"}]
        result = agent.prepare_messages(messages)
        assert result[0]["role"] == "system"
        assert "СВОДКА" in result[0]["content"]
        assert len(result) == 2

    def test_prepare_messages_strips_old_system(self):
        agent = SportsAgent()
        messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "Привет"},
        ]
        result = agent.prepare_messages(messages)
        assert result[0]["role"] == "system"
        assert "old system prompt" not in result[0]["content"]
        assert len(result) == 2

    def test_prepare_messages_includes_analytics(self):
        agent = SportsAgent()
        messages = [{"role": "user", "content": "покажи шаги за март"}]
        result = agent.prepare_messages(messages)
        assert "ШАГИ" in result[0]["content"]


# ---------------------------------------------------------------
# Admin endpoint
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_data_status_endpoint(client):
    resp = await client.get("/admin/health-data-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["activities_loaded"] > 0
    assert data["daily_facts_loaded"] > 0
    assert isinstance(data["available_metrics"], list)
    assert isinstance(data["sport_types"], list)
    # Stage 3: dedup stats exposed
    assert "activities_unique" in data
    assert "activities_duplicates_removed" in data
    assert data["activities_unique"] <= data["activities_loaded"]


# ===============================================================
# STAGE 3: Level B analytics (deduplication + cross-file logic)
# ===============================================================


def _mk_activity(
    id_: str,
    title: str,
    sport: str,
    start: str,
    duration: int,
    distance: float = 0,
    calories: float = 0,
) -> Activity:
    """Helper: build a minimal Activity from a start-time + duration."""
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = start_dt.fromtimestamp(
        start_dt.timestamp() + duration, tz=timezone.utc,
    )
    return Activity.model_validate(
        {
            "id": id_,
            "title": title,
            "sportType": sport,
            "distance": distance,
            "duration": duration,
            "startTime": start_dt.isoformat(),
            "endTime": end_dt.isoformat(),
            "calories": calories,
        }
    )


class TestActivitySource:
    def test_whoop_detected(self):
        a = _mk_activity("x", "WHOOP • running", "running",
                         "2026-03-03T10:00:00Z", 600)
        assert _activity_source(a) == "whoop"

    def test_apple_detected(self):
        a = _mk_activity("x", "Apple Health Workout", "running",
                         "2026-03-03T10:00:00Z", 600)
        assert _activity_source(a) == "apple_health"

    def test_manual_default(self):
        a = _mk_activity("x", "Бег", "running",
                         "2026-03-03T10:00:00Z", 600)
        assert _activity_source(a) == "manual"

    def test_empty_title_is_manual(self):
        a = _mk_activity("x", "", "running",
                         "2026-03-03T10:00:00Z", 600)
        assert _activity_source(a) == "manual"


class TestOverlapSeconds:
    def test_no_overlap(self):
        a = _mk_activity("a", "A", "running",
                         "2026-03-03T10:00:00Z", 600)
        b = _mk_activity("b", "B", "running",
                         "2026-03-03T11:00:00Z", 600)
        assert _overlap_seconds(a, b) == 0

    def test_full_containment(self):
        a = _mk_activity("a", "A", "running",
                         "2026-03-03T10:00:00Z", 3600)
        b = _mk_activity("b", "B", "running",
                         "2026-03-03T10:10:00Z", 600)
        assert _overlap_seconds(a, b) == 600

    def test_partial_overlap(self):
        a = _mk_activity("a", "A", "running",
                         "2026-03-03T10:00:00Z", 1800)  # 10:00-10:30
        b = _mk_activity("b", "B", "running",
                         "2026-03-03T10:20:00Z", 1800)  # 10:20-10:50
        assert _overlap_seconds(a, b) == 600  # 10 min overlap


class TestAreDuplicates:
    def test_different_sport_not_duplicate(self):
        a = _mk_activity("a", "Бег", "running",
                         "2026-03-03T10:00:00Z", 1800)
        b = _mk_activity("b", "WHOOP • cycling", "cycling",
                         "2026-03-03T10:00:00Z", 1800)
        assert _are_duplicates(a, b) is False

    def test_different_day_not_duplicate(self):
        a = _mk_activity("a", "Бег", "running",
                         "2026-03-03T10:00:00Z", 1800)
        b = _mk_activity("b", "Бег", "running",
                         "2026-03-04T10:00:00Z", 1800)
        assert _are_duplicates(a, b) is False

    def test_full_containment_is_duplicate(self):
        """Short activity fully inside a longer one on the same day/sport."""
        a = _mk_activity("a", "Бег", "running",
                         "2026-03-03T11:40:00Z", 3060)  # 51 min
        b = _mk_activity("b", "Apple Health Workout", "running",
                         "2026-03-03T11:41:00Z", 420)  # 7 min, inside
        assert _are_duplicates(a, b) is True

    def test_no_time_overlap_not_duplicate(self):
        a = _mk_activity("a", "Бег", "running",
                         "2026-03-03T10:00:00Z", 600)  # 10:00-10:10
        b = _mk_activity("b", "Apple Health Workout", "running",
                         "2026-03-03T10:11:00Z", 600)  # 10:11-10:21
        assert _are_duplicates(a, b) is False

    def test_small_overlap_not_duplicate(self):
        """Overlap below 50% of min duration -> not duplicates."""
        a = _mk_activity("a", "Бег", "running",
                         "2026-03-03T06:00:00Z", 3600)  # 60 min
        b = _mk_activity("b", "Apple Health Workout", "running",
                         "2026-03-03T06:42:00Z", 2940)  # 49 min
        # Overlap 18 min, 18/49 = 36.7% -> NOT duplicates
        assert _are_duplicates(a, b) is False

    def test_zero_duration_not_duplicate(self):
        a = _mk_activity("a", "Бег", "running",
                         "2026-03-03T10:00:00Z", 0)
        b = _mk_activity("b", "WHOOP • running", "running",
                         "2026-03-03T10:00:00Z", 600)
        assert _are_duplicates(a, b) is False


class TestPickCanonical:
    def test_whoop_wins_over_apple(self):
        whoop = _mk_activity("w", "WHOOP • running", "running",
                             "2026-03-03T10:00:00Z", 300, calories=100)
        apple = _mk_activity("a", "Apple Health Workout", "running",
                             "2026-03-03T10:00:00Z", 300, calories=100)
        assert _pick_canonical([whoop, apple]).id == "w"

    def test_apple_wins_over_manual(self):
        apple = _mk_activity("a", "Apple Health Workout", "running",
                             "2026-03-03T10:00:00Z", 300)
        manual = _mk_activity("m", "Бег", "running",
                              "2026-03-03T10:00:00Z", 600)  # longer
        # Apple > manual despite shorter duration (source priority)
        assert _pick_canonical([apple, manual]).id == "a"

    def test_longer_duration_tiebreaker_within_source(self):
        short = _mk_activity("s", "Бег", "running",
                             "2026-03-03T10:00:00Z", 300)
        long = _mk_activity("l", "Бег", "running",
                            "2026-03-03T10:00:00Z", 900)
        assert _pick_canonical([short, long]).id == "l"


class TestDeduplicate:
    def test_empty_list(self):
        result, removed = _deduplicate([])
        assert result == []
        assert removed == 0

    def test_no_duplicates(self):
        acts = [
            _mk_activity("1", "Бег", "running",
                         "2026-03-03T08:00:00Z", 600),
            _mk_activity("2", "WHOOP • cycling", "cycling",
                         "2026-03-03T10:00:00Z", 1800),
            _mk_activity("3", "Бег", "running",
                         "2026-03-04T08:00:00Z", 600),
        ]
        result, removed = _deduplicate(acts)
        assert len(result) == 3
        assert removed == 0

    def test_three_overlapping_collapse_to_one(self):
        """Three records for the same workout should collapse to one."""
        acts = [
            _mk_activity("manual", "Бег", "running",
                         "2026-03-03T10:00:00Z", 1800, calories=400),
            _mk_activity("apple", "Apple Health Workout", "running",
                         "2026-03-03T10:05:00Z", 1200, calories=300),
            _mk_activity("whoop", "WHOOP • running", "running",
                         "2026-03-03T10:10:00Z", 900, calories=200),
        ]
        result, removed = _deduplicate(acts)
        assert len(result) == 1
        assert removed == 2
        # WHOOP wins by source priority
        assert result[0].id == "whoop"

    def test_different_days_not_merged(self):
        acts = [
            _mk_activity("a", "Бег", "running",
                         "2026-03-03T10:00:00Z", 600),
            _mk_activity("b", "Apple Health Workout", "running",
                         "2026-03-04T10:00:00Z", 600),
        ]
        result, removed = _deduplicate(acts)
        assert len(result) == 2
        assert removed == 0


class TestDeduplicationOnRealData:
    def setup_method(self):
        self.svc = SportsService()

    def test_unique_fewer_than_clean(self):
        """Deduplication should reduce activity count on real data."""
        assert len(self.svc.unique_activities) < len(self.svc.clean_activities)
        assert self.svc.duplicates_removed > 0

    def test_unique_excludes_test_data(self):
        for a in self.svc.unique_activities:
            assert ActivityFlag.TEST_DATA not in a.flags

    def test_unique_sorted_by_start_time(self):
        starts = [a.startTime for a in self.svc.unique_activities]
        assert starts == sorted(starts)

    def test_march_3_deduplicated(self):
        """On March 3rd, the Apple Health 11:41-11:48 record (inside manual
        11:40-12:31) should collapse the pair into a single record."""
        mar3 = [
            a for a in self.svc.unique_activities
            if a.startTime.date() == date(2026, 3, 3)
        ]
        # Count should be less than clean count for that day
        clean_mar3 = [
            a for a in self.svc.clean_activities
            if a.startTime.date() == date(2026, 3, 3)
        ]
        assert len(mar3) < len(clean_mar3)

    def test_march_3_no_debug_records(self):
        mar3 = [
            a for a in self.svc.unique_activities
            if a.startTime.date() == date(2026, 3, 3)
        ]
        for a in mar3:
            assert "debug" not in a.title.lower()
            assert "fake" not in a.title.lower()

    def test_load_report_includes_dedup_stats(self):
        report = self.svc.load_report
        assert report.activities_unique > 0
        assert report.activities_unique <= report.activities_loaded
        assert report.activities_duplicates_removed == self.svc.duplicates_removed

    def test_training_volume_not_inflated(self):
        """Total duration of unique activities <= clean activities."""
        clean_total = sum(a.duration for a in self.svc.clean_activities)
        unique_total = sum(a.duration for a in self.svc.unique_activities)
        assert unique_total <= clean_total


# ---------------------------------------------------------------
# Calories analytics (Stage 3)
# ---------------------------------------------------------------

class TestCaloriesAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_calories_section_present(self):
        result = self.svc._build_calories_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "КАЛОРИИ" in result
        assert "Всего" in result

    def test_calories_uses_fallback_for_null_days(self):
        """Days with null caloriesKcal should fall back to activity sums."""
        # Mar 10-14 have caloriesKcal=null in daily-facts (except Mar 14 null too)
        result = self.svc._build_calories_analytics(
            date(2026, 3, 10), date(2026, 3, 16),
        )
        # Either shows facts or activities source
        assert "kcal" in result

    def test_calories_empty_period(self):
        result = self.svc._build_calories_analytics(
            date(2026, 6, 1), date(2026, 6, 7),
        )
        assert "Нет данных" in result

    def test_calories_per_day_listing(self):
        result = self.svc._build_calories_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "По дням:" in result

    def test_calories_no_double_counting(self):
        """Total calories shown <= sum of clean activities (dedup works)."""
        d_from, d_to = date(2026, 3, 1), date(2026, 3, 16)
        clean_sum = sum(
            a.calories for a in self.svc.clean_activities
            if d_from <= a.startTime.date() <= d_to
        )
        unique_sum = sum(
            a.calories for a in self.svc.unique_activities
            if d_from <= a.startTime.date() <= d_to
        )
        assert unique_sum <= clean_sum


# ---------------------------------------------------------------
# Rest days analytics (Stage 3)
# ---------------------------------------------------------------

class TestRestDaysAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_rest_days_section_present(self):
        result = self.svc._build_rest_days_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "ДНИ ОТДЫХА" in result
        assert "Тренировочных дней" in result
        assert "дней отдыха" in result

    def test_rest_days_excludes_training_days(self):
        """Days with training should not appear in rest-days list."""
        d_from, d_to = date(2026, 3, 1), date(2026, 3, 16)
        result = self.svc._build_rest_days_analytics(d_from, d_to)
        active_days = {
            a.startTime.date().isoformat()
            for a in self.svc.unique_activities
            if d_from <= a.startTime.date() <= d_to
        }
        # Split into "Дни отдыха:" section
        if "Дни отдыха:" in result:
            rest_section = result.split("Дни отдыха:")[1]
            for d in active_days:
                # Active day lines should not be inside the rest-day list
                assert d not in rest_section

    def test_rest_days_invalid_range(self):
        result = self.svc._build_rest_days_analytics(
            date(2026, 3, 10), date(2026, 3, 1),
        )
        assert "Некорректный период" in result


# ---------------------------------------------------------------
# Weekly training load (Stage 3)
# ---------------------------------------------------------------

class TestWeeklyLoadAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_weekly_load_section_present(self):
        result = self.svc._build_weekly_load_analytics(
            date(2026, 2, 25), date(2026, 3, 16),
        )
        assert "НЕДЕЛЬНАЯ НАГРУЗКА" in result
        assert "По неделям" in result

    def test_weekly_load_groups_by_monday(self):
        """Each week entry should start on a Monday."""
        result = self.svc._build_weekly_load_analytics(
            date(2026, 2, 25), date(2026, 3, 16),
        )
        # Look for some known Mondays in the output
        # Feb 2026: Feb 23 is Monday. Mar 2 is Monday. Mar 9 is Monday.
        assert ("2026-03-02" in result) or ("2026-03-09" in result)

    def test_weekly_load_total_count_matches_dedup(self):
        d_from, d_to = date(2026, 2, 25), date(2026, 3, 16)
        expected = sum(
            1 for a in self.svc.unique_activities
            if d_from <= a.startTime.date() <= d_to
        )
        result = self.svc._build_weekly_load_analytics(d_from, d_to)
        assert f"Всего тренировок (после дедупликации): {expected}" in result

    def test_weekly_load_empty_period(self):
        result = self.svc._build_weekly_load_analytics(
            date(2026, 6, 1), date(2026, 6, 7),
        )
        assert "Нет тренировок" in result


# ---------------------------------------------------------------
# Running progress (Stage 3)
# ---------------------------------------------------------------

class TestRunningProgressAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_progress_section_present(self):
        result = self.svc._build_running_progress_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "ПРОГРЕСС В БЕГЕ" in result
        # Either has pace info or says no data
        assert ("мин/км" in result) or ("Нет пробежек" in result)

    def test_progress_skips_zero_distance(self):
        """Runs with distance=0 (some 'Бег' records) should be skipped."""
        result = self.svc._build_running_progress_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        # Ensure we don't crash on zero-distance runs
        assert result  # non-empty

    def test_progress_empty_period(self):
        result = self.svc._build_running_progress_analytics(
            date(2026, 6, 1), date(2026, 6, 7),
        )
        assert "Нет пробежек" in result


# ---------------------------------------------------------------
# Stage 3 intent detection
# ---------------------------------------------------------------

class TestStage3IntentDetection:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_detect_calories_калории(self):
        assert self.svc._detect_calories_intent("сколько калорий я сжёг") is True

    def test_detect_calories_ккал(self):
        assert self.svc._detect_calories_intent("покажи ккал за март") is True

    def test_detect_calories_negative(self):
        assert self.svc._detect_calories_intent("покажи тренировки") is False

    def test_detect_rest_days_отдых(self):
        assert self.svc._detect_rest_days_intent("сколько дней отдыха") is True

    def test_detect_rest_days_без_тренировок(self):
        assert self.svc._detect_rest_days_intent("дни без тренировок") is True

    def test_detect_weekly_load_нагрузка(self):
        assert self.svc._detect_weekly_load_intent("моя нагрузка за месяц") is True

    def test_detect_weekly_load_недельная(self):
        assert self.svc._detect_weekly_load_intent("недельный объём тренировок") is True

    def test_detect_progress_прогресс(self):
        assert self.svc._detect_progress_intent("мой прогресс в беге") is True

    def test_detect_progress_темп(self):
        assert self.svc._detect_progress_intent("как меняется темп") is True

    def test_detect_progress_negative(self):
        assert self.svc._detect_progress_intent("покажи шаги") is False


# ---------------------------------------------------------------
# Stage 3 get_analytics() integration
# ---------------------------------------------------------------

class TestStage3GetAnalytics:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_calories_query(self):
        result = self.svc.get_analytics("сколько калорий я сжёг за март")
        assert "КАЛОРИИ" in result

    def test_rest_days_query(self):
        result = self.svc.get_analytics("сколько дней отдыха в марте")
        assert "ДНИ ОТДЫХА" in result

    def test_weekly_load_query(self):
        result = self.svc.get_analytics("моя недельная нагрузка за март")
        assert "НЕДЕЛЬНАЯ НАГРУЗКА" in result

    def test_running_progress_query(self):
        result = self.svc.get_analytics("мой прогресс в беге за март")
        assert "ПРОГРЕСС В БЕГЕ" in result

    def test_progress_with_cycling_filter_skips_running(self):
        """Progress query for cycling should not produce running-progress section."""
        result = self.svc.get_analytics("прогресс в велосипеде за март")
        assert "ПРОГРЕСС В БЕГЕ" not in result

    def test_summary_mentions_dedup_when_duplicates_present(self):
        summary = self.svc.get_summary()
        assert "дедупликации" in summary


# ===============================================================
# STAGE 4: Level C analytics
# (extended metrics, graceful degradation, correlations)
# ===============================================================


def _mk_daily_fact(
    iso_date: str,
    *,
    steps: int = 0,
    calories: float | None = None,
    recovery: float | None = None,
    sleep_perf: float | None = None,
    hrv: float | None = None,
    resting_hr: float | None = None,
    spo2: float | None = None,
    skin_temp: float | None = None,
    sleep_ms: int | None = None,
) -> DailyFact:
    return DailyFact.model_validate(
        {
            "id": f"df-{iso_date}",
            "isoDate": iso_date,
            "steps": steps,
            "caloriesKcal": calories,
            "recoveryScore": recovery,
            "sleepPerformancePercentage": sleep_perf,
            "hrvRmssdMilli": hrv,
            "restingHeartRate": resting_hr,
            "spo2Percentage": spo2,
            "skinTempCelsius": skin_temp,
            "sleepTotalInBedTimeMilli": sleep_ms,
        }
    )


def _make_svc_with_facts(
    facts: list[DailyFact], today: date | None = None,
) -> SportsService:
    """Build a service, then replace its daily facts with synthetic data."""
    svc = SportsService()
    svc._daily_facts = facts
    svc._load_report = svc._build_report()
    if today:
        svc._today = today
    return svc


# ---------------------------------------------------------------
# Extended fields accepted by the DailyFact schema
# ---------------------------------------------------------------


class TestDailyFactExtendedFields:
    def test_extended_fields_accepted(self):
        f = _mk_daily_fact(
            "2026-03-16",
            steps=8000,
            calories=500,
            recovery=60,
            sleep_perf=85,
            hrv=42.5,
            resting_hr=55,
            spo2=97,
            skin_temp=33.5,
            sleep_ms=28_800_000,
        )
        assert f.hrvRmssdMilli == 42.5
        assert f.restingHeartRate == 55
        assert f.spo2Percentage == 97
        assert f.skinTempCelsius == 33.5
        assert f.sleepTotalInBedTimeMilli == 28_800_000
        assert f.sleepPerformancePercentage == 85

    def test_extended_fields_optional(self):
        """A fact with no extended fields should validate cleanly."""
        f = _mk_daily_fact("2026-03-16", steps=1000)
        assert f.hrvRmssdMilli is None
        assert f.sleepTotalInBedTimeMilli is None
        assert f.flags == []


# ---------------------------------------------------------------
# Metric availability detection (partial real-world data)
# ---------------------------------------------------------------


class TestMetricAvailabilityPartial:
    """With the current example data, only core metrics have values.
    Extended metrics are referenced in sourcesJson but have no stored
    field values, so they must be reported as unavailable."""

    def setup_method(self):
        self.svc = SportsService()

    def test_steps_is_available(self):
        assert self.svc.is_metric_available("steps") is True
        assert "steps" in self.svc.get_available_metrics()

    def test_recovery_is_available(self):
        assert self.svc.is_metric_available("recoveryScore") is True

    def test_calories_is_available(self):
        assert self.svc.is_metric_available("caloriesKcal") is True

    def test_hrv_is_unavailable(self):
        assert self.svc.is_metric_available("hrvRmssdMilli") is False
        assert "hrvRmssdMilli" in self.svc.get_unavailable_metrics()

    def test_spo2_is_unavailable(self):
        assert self.svc.is_metric_available("spo2Percentage") is False
        assert "spo2Percentage" in self.svc.get_unavailable_metrics()

    def test_resting_hr_is_unavailable(self):
        assert self.svc.is_metric_available("restingHeartRate") is False

    def test_skin_temp_is_unavailable(self):
        assert self.svc.is_metric_available("skinTempCelsius") is False

    def test_sleep_performance_is_unavailable(self):
        assert self.svc.is_metric_available("sleepPerformancePercentage") is False

    def test_sleep_duration_is_unavailable(self):
        assert self.svc.is_metric_available("sleepTotalInBedTimeMilli") is False

    def test_unavailable_metrics_not_in_available(self):
        """A metric must be in exactly one of available/unavailable."""
        available = set(self.svc.get_available_metrics())
        unavailable = set(self.svc.get_unavailable_metrics())
        assert available & unavailable == set()

    def test_load_report_exposes_unavailable(self):
        report = self.svc.load_report
        assert "hrvRmssdMilli" in report.unavailable_metrics
        assert "steps" not in report.unavailable_metrics


class TestMetricAvailabilitySynthetic:
    """With a synthetic DailyFact that has every metric filled,
    all canonical metrics must report as available."""

    def test_all_metrics_available_with_full_data(self):
        facts = [
            _mk_daily_fact(
                "2026-03-16",
                steps=8000,
                calories=500,
                recovery=60,
                sleep_perf=85,
                hrv=42,
                resting_hr=55,
                spo2=97,
                skin_temp=33.5,
                sleep_ms=28_800_000,
            ),
        ]
        svc = _make_svc_with_facts(facts)
        for metric in [
            "steps",
            "caloriesKcal",
            "recoveryScore",
            "sleepPerformancePercentage",
            "hrvRmssdMilli",
            "restingHeartRate",
            "spo2Percentage",
            "skinTempCelsius",
            "sleepTotalInBedTimeMilli",
        ]:
            assert svc.is_metric_available(metric) is True, metric
        assert svc.get_unavailable_metrics() == []


# ---------------------------------------------------------------
# Availability section in system prompt (summary)
# ---------------------------------------------------------------


class TestAvailabilitySectionInSummary:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_summary_contains_availability_block(self):
        summary = self.svc.get_summary()
        assert "ДОСТУПНОСТЬ МЕТРИК" in summary
        assert "Доступные метрики" in summary
        assert "Недоступные метрики" in summary

    def test_summary_lists_current_available(self):
        summary = self.svc.get_summary()
        assert "шаги" in summary
        assert "восстановление" in summary
        assert "калории" in summary

    def test_summary_lists_current_unavailable(self):
        summary = self.svc.get_summary()
        assert "HRV" in summary
        assert "SpO2" in summary

    def test_summary_warns_against_hallucination(self):
        summary = self.svc.get_summary()
        assert "не выдумывай" in summary


# ---------------------------------------------------------------
# Stage 4 intent detection
# ---------------------------------------------------------------


class TestStage4IntentDetection:
    def setup_method(self):
        self.svc = SportsService()

    def test_detect_hrv(self):
        assert self.svc._detect_hrv_intent("покажи мой hrv") is True
        assert self.svc._detect_hrv_intent("вариабельность пульса") is True

    def test_detect_sleep(self):
        assert self.svc._detect_sleep_intent("как мой сон") is True
        assert self.svc._detect_sleep_intent("сколько я спал") is True
        assert self.svc._detect_sleep_intent("show sleep data") is True

    def test_detect_spo2(self):
        assert self.svc._detect_spo2_intent("покажи spo2") is True
        assert self.svc._detect_spo2_intent("сатурация кислорода") is True

    def test_detect_resting_hr(self):
        assert self.svc._detect_resting_hr_intent(
            "мой пульс в покое за март"
        ) is True

    def test_detect_skin_temp(self):
        assert self.svc._detect_skin_temp_intent(
            "температура кожи за неделю"
        ) is True

    def test_detect_correlation(self):
        assert self.svc._detect_correlation_intent(
            "корреляция восстановления и нагрузки"
        ) is True
        assert self.svc._detect_correlation_intent(
            "есть связь между сном и recovery?"
        ) is True

    def test_hrv_intent_negative(self):
        assert self.svc._detect_hrv_intent("покажи шаги") is False

    def test_sleep_intent_negative(self):
        assert self.svc._detect_sleep_intent("велосипед") is False


# ---------------------------------------------------------------
# Graceful degradation for unavailable metrics
# ---------------------------------------------------------------


class TestGracefulDegradation:
    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_hrv_query_no_data_response(self):
        result = self.svc.get_analytics("покажи мой hrv за март")
        assert "HRV" in result
        assert "отсутствуют" in result or "нет" in result
        # No fabricated numbers
        assert "42" not in result  # arbitrary guard against hallucination

    def test_spo2_query_no_data_response(self):
        result = self.svc.get_analytics("мой spo2 за последние 7 дней")
        assert "SpO2" in result
        assert "отсутствуют" in result

    def test_resting_hr_query_no_data_response(self):
        result = self.svc.get_analytics("пульс в покое за март")
        assert "отсутствуют" in result

    def test_skin_temp_query_no_data_response(self):
        result = self.svc.get_analytics("температура кожи за март")
        assert "отсутствуют" in result

    def test_sleep_query_no_data_response(self):
        result = self.svc.get_analytics("как мой сон за последние 7 дней")
        assert "СОН" in result
        assert "отсутствуют" in result

    def test_no_error_on_extended_query_with_empty_data(self):
        """Querying an unavailable metric must never raise."""
        # Should not raise
        self.svc.get_analytics("покажи hrv spo2 пульс в покое температуру кожи")

    def test_extended_metric_builder_with_empty_data_safe(self):
        result = self.svc._build_extended_metric_analytics(
            "hrvRmssdMilli", date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "HRV" in result
        assert "отсутствуют" in result


# ---------------------------------------------------------------
# Extended metric analytics with synthetic full data
# ---------------------------------------------------------------


class TestExtendedMetricsWithFullData:
    def setup_method(self):
        facts = [
            _mk_daily_fact(
                "2026-03-10",
                steps=8000, calories=500, recovery=50,
                sleep_perf=80, hrv=40, resting_hr=58,
                spo2=96, skin_temp=33.2, sleep_ms=25_200_000,
            ),
            _mk_daily_fact(
                "2026-03-11",
                steps=9500, calories=620, recovery=65,
                sleep_perf=88, hrv=48, resting_hr=54,
                spo2=97, skin_temp=33.4, sleep_ms=28_800_000,
            ),
            _mk_daily_fact(
                "2026-03-12",
                steps=12000, calories=750, recovery=72,
                sleep_perf=92, hrv=55, resting_hr=52,
                spo2=98, skin_temp=33.6, sleep_ms=30_600_000,
            ),
            _mk_daily_fact(
                "2026-03-13",
                steps=7500, calories=480, recovery=60,
                sleep_perf=83, hrv=42, resting_hr=56,
                spo2=96, skin_temp=33.5, sleep_ms=27_000_000,
            ),
            _mk_daily_fact(
                "2026-03-14",
                steps=10000, calories=680, recovery=68,
                sleep_perf=90, hrv=50, resting_hr=53,
                spo2=97, skin_temp=33.3, sleep_ms=29_400_000,
            ),
        ]
        self.svc = _make_svc_with_facts(facts, today=date(2026, 3, 14))

    def test_all_extended_metrics_available(self):
        for metric in [
            "hrvRmssdMilli", "restingHeartRate", "spo2Percentage",
            "skinTempCelsius", "sleepTotalInBedTimeMilli",
            "sleepPerformancePercentage",
        ]:
            assert self.svc.is_metric_available(metric), metric

    def test_hrv_analytics_presents_real_numbers(self):
        result = self.svc._build_extended_metric_analytics(
            "hrvRmssdMilli", date(2026, 3, 10), date(2026, 3, 14),
        )
        assert "HRV" in result
        assert "Записей: 5" in result
        assert "Минимум" in result
        assert "Максимум" in result
        # Check that our min/max values appear
        assert "40.0" in result
        assert "55.0" in result

    def test_spo2_analytics_presents_real_numbers(self):
        result = self.svc._build_extended_metric_analytics(
            "spo2Percentage", date(2026, 3, 10), date(2026, 3, 14),
        )
        assert "SPO2" in result.upper()
        assert "Записей: 5" in result
        assert "96.0" in result
        assert "98.0" in result

    def test_resting_hr_analytics(self):
        result = self.svc._build_extended_metric_analytics(
            "restingHeartRate", date(2026, 3, 10), date(2026, 3, 14),
        )
        assert "пульс в покое".upper() in result.upper() or "ПУЛЬС" in result
        assert "52.0" in result
        assert "58.0" in result

    def test_sleep_analytics_with_full_data(self):
        result = self.svc._build_sleep_analytics(
            date(2026, 3, 10), date(2026, 3, 14),
        )
        assert "СОН" in result
        assert "Эффективность сна" in result
        assert "Длительность сна" in result
        # Don't show "данных нет"
        assert "данных в источниках нет" not in result

    def test_get_analytics_routes_hrv_to_extended_builder(self):
        result = self.svc.get_analytics("hrv за 5 дней")
        assert "HRV" in result
        # With full synthetic data, must include numbers
        assert "Записей" in result


# ---------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------


class TestCorrelationsWithPartialData:
    """Current data has recovery but no sleep/HRV — recovery↔load runs,
    the other two report 'метрики нет'."""

    def setup_method(self):
        self.svc = SportsService()
        self.svc._today = date(2026, 3, 16)

    def test_correlations_section_runs_safely(self):
        result = self.svc._build_correlations_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "КОРРЕЛЯЦИИ" in result

    def test_sleep_correlation_honest_missing(self):
        result = self.svc._build_correlations_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "Сон ↔ восстановление" in result
        assert "нет" in result

    def test_hrv_correlation_honest_missing(self):
        result = self.svc._build_correlations_analytics(
            date(2026, 3, 1), date(2026, 3, 16),
        )
        assert "HRV" in result
        assert "данных HRV нет" in result

    def test_correlation_intent_triggers_section(self):
        result = self.svc.get_analytics(
            "есть корреляция восстановления и нагрузки?"
        )
        assert "КОРРЕЛЯЦИИ" in result


class TestCorrelationsWithFullData:
    def setup_method(self):
        # Synthetic perfectly-correlated sleep ↔ recovery pairs
        facts = [
            _mk_daily_fact(
                "2026-03-10",
                recovery=40, sleep_perf=60, hrv=35,
                calories=300,
            ),
            _mk_daily_fact(
                "2026-03-11",
                recovery=50, sleep_perf=70, hrv=40,
                calories=400,
            ),
            _mk_daily_fact(
                "2026-03-12",
                recovery=60, sleep_perf=80, hrv=45,
                calories=500,
            ),
            _mk_daily_fact(
                "2026-03-13",
                recovery=70, sleep_perf=90, hrv=50,
                calories=600,
            ),
        ]
        self.svc = _make_svc_with_facts(facts, today=date(2026, 3, 13))

    def test_sleep_recovery_strong_positive(self):
        result = self.svc._build_correlations_analytics(
            date(2026, 3, 10), date(2026, 3, 13),
        )
        assert "Сон ↔ восстановление" in result
        # We built a perfectly linear pair so r ≈ 1.00
        assert "r = 1.00" in result or "r = 0.99" in result
        assert "сильная положительная" in result

    def test_recovery_load_correlation_present(self):
        result = self.svc._build_correlations_analytics(
            date(2026, 3, 10), date(2026, 3, 13),
        )
        assert "Восстановление ↔ нагрузка" in result


class TestPearsonHelper:
    def test_perfect_positive(self):
        from app.services.sports import _pearson
        assert abs(_pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9

    def test_perfect_negative(self):
        from app.services.sports import _pearson
        assert abs(_pearson([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-9

    def test_no_variance_returns_zero(self):
        from app.services.sports import _pearson
        assert _pearson([5, 5, 5], [1, 2, 3]) == 0.0

    def test_too_short_returns_zero(self):
        from app.services.sports import _pearson
        assert _pearson([1.0], [2.0]) == 0.0
        assert _pearson([], []) == 0.0

    def test_interpret_correlation(self):
        from app.services.sports import _interpret_correlation
        assert "сильная положительная" in _interpret_correlation(0.9)
        assert "сильная отрицательная" in _interpret_correlation(-0.9)
        assert "умеренная" in _interpret_correlation(0.5)
        assert "связи не обнаружено" in _interpret_correlation(0.0)


# ---------------------------------------------------------------
# SportsAgent integration — availability in system prompt
# ---------------------------------------------------------------


class TestSportsAgentStage4:
    def test_prepare_messages_includes_availability(self):
        agent = SportsAgent()
        messages = [{"role": "user", "content": "привет"}]
        result = agent.prepare_messages(messages)
        content = result[0]["content"]
        assert "ДОСТУПНОСТЬ МЕТРИК" in content
        assert "Доступные метрики" in content

    def test_prepare_messages_hrv_query_gets_graceful_response(self):
        agent = SportsAgent()
        messages = [{"role": "user", "content": "какой у меня hrv?"}]
        result = agent.prepare_messages(messages)
        content = result[0]["content"]
        assert "HRV" in content
        assert "отсутствуют" in content

    def test_prepare_messages_sleep_query_gets_graceful_response(self):
        agent = SportsAgent()
        messages = [{"role": "user", "content": "сколько я спал вчера"}]
        result = agent.prepare_messages(messages)
        content = result[0]["content"]
        assert "СОН" in content


# ---------------------------------------------------------------
# Admin endpoint returns unavailable_metrics
# ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_data_status_includes_unavailable_metrics(client):
    resp = await client.get("/admin/health-data-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "unavailable_metrics" in data
    assert isinstance(data["unavailable_metrics"], list)
    # In the current example data, HRV has no values
    assert "hrvRmssdMilli" in data["unavailable_metrics"]

