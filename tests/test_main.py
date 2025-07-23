import types
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import Model.__main__ as main


class DummyLearningResults:
    stations_analyzed = 42
    track_segments = 87
    railyards_found = 3
    train_services_analyzed = 120
    data_quality_score = 0.91
    coverage_completeness = 0.88
    learning_confidence = 0.86
    model_accuracies = {"baseline": 0.95}
    station_patterns = {}
    track_patterns = {}
    key_insights = ["Insight A", "Insight B"]
    recommendations = ["Rec 1", "Rec 2", "Rec 3"]
    country = "Belgium"
    learning_date = "2025-07-23"
    learning_time = 1.3


class DummyPlan:
    name = "demo_route"
    total_length_km = 312.4
    stations = ["A", "B", "C"]
    total_cost = 1_200_000


class DummyGradeReport:
    overall_score = 87.4
    cost_score = 82.0
    feasibility_score = 90.0
    environmental_score = 88.0


@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "outputs"


@pytest.fixture
def dummy_config():
    dummy = types.SimpleNamespace()
    dummy.paths = types.SimpleNamespace(output_dir=Path("."))
    dummy.logging = types.SimpleNamespace(level=main.LogLevel.INFO)
    return dummy


def test_validate_arguments_errors():
    args = main.argparse.Namespace(mode="learn", country=None, config=None, input=None, plan=None, scenario=None, test_suite=None, benchmark=False)
    assert main.validate_arguments(args) is False
    args.mode = "generate"
    assert main.validate_arguments(args) is False
    args.mode = "test"
    assert main.validate_arguments(args) is False
    args.mode = "grade"
    assert main.validate_arguments(args) is False


def _namespace(**kwargs):
    d = {
        "config": None,
        "output_dir": str(Path.cwd() / "tests_out"),
        "log_level": "info",
        "log_file": None,
        "verbose": False,
        "country": "Belgium",
        "train_types": "IC,ICE",
        "focus": None,
        "data_sources": None,
        "input": "cities.csv",
        "optimize": "cost",
        "constraints": None,
        "route_name": None,
        "scenario": "alpine_crossing",
        "test_suite": None,
        "benchmark": False,
        "plan": "demo_plan.json",
        "metrics": "all",
        "reference": None,
    }
    d.update(kwargs)
    return main.argparse.Namespace(**d)


def test_execute_learn_mode_success(tmp_output_dir, dummy_config, monkeypatch):
    args = _namespace(mode="learn", output_dir=str(tmp_output_dir))
    fake_learner = mock.MagicMock()
    fake_learner.execute.return_value = DummyLearningResults()
    monkeypatch.setattr(main, "RailwayLearner", mock.MagicMock(return_value=fake_learner))
    status = main.execute_learn_mode(args, dummy_config)
    assert status == 0
    expected_model_dir = tmp_output_dir / "models" / args.country.lower()
    assert expected_model_dir.exists()
    assert (expected_model_dir / "learning_summary.txt").exists()
    fake_learner.save_models.assert_called_once()


def test_execute_generate_mode_success(tmp_output_dir, dummy_config, monkeypatch):
    args = _namespace(mode="generate", output_dir=str(tmp_output_dir), input="cities.csv")
    fake_generator = mock.MagicMock()
    fake_generator.create_plan.return_value = DummyPlan()

    def _touch(path):
        Path(path).touch()

    fake_generator.save_plan.side_effect = _touch
    monkeypatch.setattr(main, "RouteGenerator", mock.MagicMock(return_value=fake_generator))
    status = main.execute_generate_mode(args, dummy_config)
    assert status == 0
    assert any(p.suffix == ".json" for p in tmp_output_dir.iterdir())
    fake_generator.save_plan.assert_called_once()


def test_execute_test_mode_success(dummy_config, monkeypatch):
    args = _namespace(mode="test", benchmark=True)
    fake_tester = mock.MagicMock()
    fake_tester.run_benchmark.return_value = {"passed": 10, "failed": 0, "success_rate": 1.0}
    monkeypatch.setattr(main, "ScenarioTester", mock.MagicMock(return_value=fake_tester))
    status = main.execute_test_mode(args, dummy_config)
    assert status == 0
    fake_tester.run_benchmark.assert_called_once()


def test_execute_grade_mode_success(tmp_output_dir, dummy_config, monkeypatch):
    args = _namespace(mode="grade", output_dir=str(tmp_output_dir), plan="demo_plan.json")
    fake_grader = mock.MagicMock()
    fake_grader.evaluate_plan.return_value = DummyGradeReport()

    def _touch(path):
        Path(path).touch()

    fake_grader.save_report.side_effect = _touch
    monkeypatch.setattr(main, "PlanGrader", mock.MagicMock(return_value=fake_grader))
    status = main.execute_grade_mode(args, dummy_config)
    assert status == 0
    assert any(p.name.startswith("grade_report_") for p in tmp_output_dir.iterdir())
    fake_grader.save_report.assert_called_once()
