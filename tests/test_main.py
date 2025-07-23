# tests/test_main.py
import types
from pathlib import Path
from unittest import mock

import pytest

# Import the entry‑point module
import Model.__main__ as main


# --------------------------------------------------------------------------- #
#                   ----------  Shared dummy objects  ----------               #
# --------------------------------------------------------------------------- #
class DummyLearningResults:
    # attributes accessed in execute_learn_mode
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
    recommendations = ["Rec 1", "Rec 2", "Rec 3"]
    country = "Belgium"
    learning_date = "2025‑07‑23"
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
    """Returns a temporary directory path for --output-dir."""
    return tmp_path / "outputs"


@pytest.fixture
def dummy_config():
    """Minimal stub for RailwayConfig; only the bits used in __main__."""
    dummy = types.SimpleNamespace()
    dummy.paths = types.SimpleNamespace(output_dir=Path("."))  # will be overridden
    dummy.logging = types.SimpleNamespace(level=main.LogLevel.INFO)
    return dummy


# --------------------------------------------------------------------------- #
#                              Argument parsing                               #
# --------------------------------------------------------------------------- #
def test_validate_arguments_errors():
    # Missing --country for learn
    args = main.argparse.Namespace(
        mode="learn",
        country=None,
        config=None,
        input=None,
        plan=None,
        scenario=None,
        test_suite=None,
        benchmark=False,
    )
    assert main.validate_arguments(args) is False

    # Missing --input for generate
    args.mode = "generate"
    assert main.validate_arguments(args) is False

    # Test mode with no scenario/suite/benchmark
    args.mode = "test"
    assert main.validate_arguments(args) is False

    # Grade mode without --plan
    args.mode = "grade"
    assert main.validate_arguments(args) is False


# --------------------------------------------------------------------------- #
#                             Execution modes (OK)                            #
# --------------------------------------------------------------------------- #
def _namespace(**kwargs):
    """Convenience helper to build argparse.Namespace objects."""
    d = {
        "config": None,
        "output_dir": str(Path.cwd() / "tests_out"),
        "log_level": "info",
        "log_file": None,
        "verbose": False,
        # learn‑specific
        "country": "Belgium",
        "train_types": "IC,ICE",
        "focus": None,
        "data_sources": None,
        # generate‑specific
        "input": "cities.csv",
        "optimize": "cost",
        "constraints": None,
        "route_name": None,
        # test‑specific
        "scenario": "alpine_crossing",
        "test_suite": None,
        "benchmark": False,
        # grade‑specific
        "plan": "demo_plan.json",
        "metrics": "all",
        "reference": None,
    }
    d.update(kwargs)
    return main.argparse.Namespace(**d)


def test_execute_learn_mode_success(tmp_output_dir, dummy_config, monkeypatch):
    args = _namespace(mode="learn", output_dir=str(tmp_output_dir))

    # Patch RailwayLearner
    fake_learner = mock.MagicMock()
    fake_learner.execute.return_value = DummyLearningResults()
    monkeypatch.setattr(main, "RailwayLearner", mock.MagicMock(return_value=fake_learner))

    status = main.execute_learn_mode(args, dummy_config)
    assert status == 0
    # model directory created?
    expected_model_dir = tmp_output_dir / "models" / args.country.lower()
    assert expected_model_dir.exists()
    # summary file saved?
    assert (expected_model_dir / "learning_summary.txt").exists()
    fake_learner.save_models.assert_called_once()


def test_execute_generate_mode_success(tmp_output_dir, dummy_config, monkeypatch):
    args = _namespace(mode="generate", output_dir=str(tmp_output_dir), input="cities.csv")

    # Patch RouteGenerator
    fake_generator = mock.MagicMock()
    fake_generator.create_plan.return_value = DummyPlan()
    monkeypatch.setattr(main, "RouteGenerator", mock.MagicMock(return_value=fake_generator))

    status = main.execute_generate_mode(args, dummy_config)
    assert status == 0
    # A plan file should have been written to the output dir
    assert any(p.suffix == ".json" for p in tmp_output_dir.iterdir())
    fake_generator.save_plan.assert_called_once()


def test_execute_test_mode_success(dummy_config, monkeypatch):
    args = _namespace(mode="test", benchmark=True)  # simplest path

    # Patch ScenarioTester
    fake_tester = mock.MagicMock()
    fake_tester.run_benchmark.return_value = {"passed": 10, "failed": 0, "success_rate": 1.0}
    monkeypatch.setattr(main, "ScenarioTester", mock.MagicMock(return_value=fake_tester))

    status = main.execute_test_mode(args, dummy_config)
    assert status == 0
    fake_tester.run_benchmark.assert_called_once()


def test_execute_grade_mode_success(tmp_output_dir, dummy_config, monkeypatch):
    args = _namespace(mode="grade", output_dir=str(tmp_output_dir), plan="demo_plan.json")

    # Patch PlanGrader
    fake_grader = mock.MagicMock()
    fake_grader.evaluate_plan.return_value = DummyGradeReport()
    monkeypatch.setattr(main, "PlanGrader", mock.MagicMock(return_value=fake_grader))

    status = main.execute_grade_mode(args, dummy_config)
    assert status == 0
    # A report file should have been saved
    assert any(p.name.startswith("grade_report_") for p in tmp_output_dir.iterdir())
    fake_grader.save_report.assert_called_once()
