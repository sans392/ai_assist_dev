"""Tests for Sports Assistant — Stage 1: data loading, validation, agent skeleton."""

import pytest
from pydantic import ValidationError

from app.agents import get_agent_by_mode
from app.agents.sports import SportsAgent
from app.models.health_schemas import Activity, ActivityFlag, DailyFact, HealthDay
from app.services.sports import SportsService


# ---------------------------------------------------------------
# Pydantic model validation
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
# SportsService — data loading
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
