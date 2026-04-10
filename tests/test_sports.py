"""Tests for Sports Assistant — Stage 1 + Stage 2 Level A analytics."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.agents import get_agent_by_mode
from app.agents.sports import SportsAgent
from app.models.health_schemas import Activity, ActivityFlag, DailyFact, HealthDay
from app.services.sports import SportsService, _compute_trend


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
