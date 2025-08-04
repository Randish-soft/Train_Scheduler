"""
pipeline/pipeline_initiator.py
Orchestrates the train-plotter pipeline end-to-end.

Usage (via python -m pipeline):
    python -m pipeline pipeline_initiator run --country Belgium --mode learn
    python -m pipeline pipeline_initiator run --country Lebanon --mode infer --budget 2e9 --ref Belgium
"""
from __future__ import annotations

import importlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("pipeline_initiator")
logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# Shared context object (dict-like)
# -----------------------------------------------------------------------------
class Ctx(Dict[str, Any]):
    """
    Thin wrapper so we can attach helpers or type‐hints later.
    Example: ctx["country"], ctx["terrain"], ctx["routes"], ...
    """

    def dump(self) -> str:
        return json.dumps(self, indent=2, default=str)

# -----------------------------------------------------------------------------
# Base step
# -----------------------------------------------------------------------------
class Step:
    """
    Every pipeline block extends this.  Override `.run(ctx)`.

    Subclasses can optionally implement:
        • uses  – list of keys they expect in ctx (for debugging)
        • adds  – list of keys they add to ctx (for debugging)
    """

    name: str = "base-step"
    uses: List[str] = []
    adds: List[str] = []

    def __call__(self, ctx: Ctx) -> Ctx:  # noqa: D401
        logger.info("▶  %s", self.name)
        return self.run(ctx)

    # --------------------------------------------------------------------- #
    # OVERRIDE THIS METHOD IN SUBCLASSES
    # --------------------------------------------------------------------- #
    def run(self, ctx: Ctx) -> Ctx:
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    # --------------------------------------------------------------------- #
    # Utility: try calling real module if it exists, else stub
    # --------------------------------------------------------------------- #
    def _maybe_call(self, module_path: str, func: str, *args, **kwargs):
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func)
            return fn(*args, **kwargs)
        except (ImportError, AttributeError) as exc:
            logger.debug("    ↪ %s.%s not found – %s", module_path, func, exc)
            return None


# -----------------------------------------------------------------------------
# Concrete steps – mirror your diagram
# Only minimal logic/stubs here; real work lives in pipeline.core.*, etc.
# -----------------------------------------------------------------------------
class Initialize(Step):
    name = "Initialize"
    adds = ["run_id", "start_ts"]

    def run(self, ctx: Ctx) -> Ctx:
        ctx["run_id"] = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        ctx["start_ts"] = datetime.utcnow().isoformat()
        return ctx


class ProcessingInput(Step):
    name = "Processing Input"
    uses = ["country"]
    adds = ["raw_inputs"]

    def run(self, ctx: Ctx) -> Ctx:
        out = self._maybe_call("pipeline.core.data.loaders", "load_inputs", ctx["country"])
        ctx["raw_inputs"] = out or {}
        return ctx


class ProcessingTerrain(Step):
    name = "Processing Terrain"
    uses = ["country"]
    adds = ["terrain"]

    def run(self, ctx: Ctx) -> Ctx:
        terrain = self._maybe_call("pipeline.core.data.loaders", "load_terrain", ctx["country"])
        ctx["terrain"] = terrain
        return ctx


class ReferencingSimilarCountry(Step):
    name = "Referencing Similar Country"
    uses = ["country"]
    adds = ["reference_country", "reference_artifacts"]

    def run(self, ctx: Ctx) -> Ctx:
        # If user already supplied --ref, keep it
        ref = ctx.get("reference_country")
        if ref is None:
            ref = self._maybe_call("pipeline.core.data.artifacts", "find_similar_reference", ctx["country"])
        ctx["reference_country"] = ref
        ctx["reference_artifacts"] = (
            self._maybe_call("pipeline.core.data.artifacts", "load_country_artifacts", ref) if ref else {}
        )
        return ctx


class ProcessingDemand(Step):
    name = "Processing Demand"
    uses = ["raw_inputs"]
    adds = ["demand_model"]

    def run(self, ctx: Ctx) -> Ctx:
        model = self._maybe_call("pipeline.core.processing.demand", "estimate_demand", ctx["raw_inputs"])
        ctx["demand_model"] = model
        return ctx


class ConstraintAnalyzer(Step):
    name = "Constraint Analyzer"
    uses = ["terrain", "demand_model"]
    adds = ["constraints"]

    def run(self, ctx: Ctx) -> Ctx:
        constraints = self._maybe_call(
        "pipeline.core.optimization.constraint_analyzer",
        "analyze",
        ctx["terrain"],
        ctx["demand_model"],
        country=ctx["country"]
    )


        ctx["constraints"] = constraints
        return ctx


class PlottingRoute(Step):
    name = "Plotting Route"
    uses = ["terrain", "constraints"]
    adds = ["routes"]

    def run(self, ctx: Ctx) -> Ctx:
        routes = self._maybe_call(
            "pipeline.core.processing.plotting_route", "plot", ctx["terrain"], ctx["constraints"]
        )
        ctx["routes"] = routes
        return ctx


class NIMBYAnalyzer(Step):
    name = "NIMBY Analyzer"
    uses = ["routes"]
    adds = ["nimby_constraints"]

    def run(self, ctx: Ctx) -> Ctx:
        nimby = self._maybe_call("pipeline.core.processing.nimby_analyzer", "analyze", ctx["routes"])
        ctx["nimby_constraints"] = nimby
        return ctx


class RouteOptimizer(Step):
    name = "Route Optimizer"
    uses = ["routes", "nimby_constraints"]
    adds = ["optimized_routes"]

    def run(self, ctx: Ctx) -> Ctx:
        opt_routes = self._maybe_call(
            "pipeline.core.optimization.optimising_route", "optimize", ctx["routes"], ctx["nimby_constraints"]
        )
        ctx["optimized_routes"] = opt_routes
        return ctx


class ChoosingTrainForRoute(Step):
    name = "Choosing Train For Route"
    uses = ["optimized_routes"]
    adds = ["fleet_assignment"]

    def run(self, ctx: Ctx) -> Ctx:
        fleet = self._maybe_call("pipeline.core.processing.choosing_train_for_route", "choose", ctx["optimized_routes"])
        ctx["fleet_assignment"] = fleet
        return ctx


class CreatingTimeTable(Step):
    name = "Creating Time Table"
    uses = ["optimized_routes", "fleet_assignment", "demand_model"]
    adds = ["timetable"]

    def run(self, ctx: Ctx) -> Ctx:
        tt = self._maybe_call(
            "pipeline.core.processing.creating_time_table",
            "build",
            ctx["optimized_routes"],
            ctx["fleet_assignment"],
            ctx["demand_model"],
        )
        ctx["timetable"] = tt
        return ctx


class OptimizingTimeTable(Step):
    name = "Optimizing Time Table"
    uses = ["timetable", "constraints"]
    adds = ["optimized_timetable"]

    def run(self, ctx: Ctx) -> Ctx:
        opt_tt = self._maybe_call(
            "pipeline.core.optimization.optimising_time_table", "optimize", ctx["timetable"], ctx["constraints"]
        )
        ctx["optimized_timetable"] = opt_tt
        return ctx


class RailyardPlotter(Step):
    name = "Railyard Plotter"
    uses = ["optimized_routes"]
    adds = ["railyard_layout"]

    def run(self, ctx: Ctx) -> Ctx:
        layout = self._maybe_call("pipeline.core.processing.railyard_plotter", "plot", ctx["optimized_routes"])
        ctx["railyard_layout"] = layout
        return ctx


class RailyardOptimizer(Step):
    name = "Railyard Optimizer"
    uses = ["railyard_layout", "optimized_timetable"]
    adds = ["optimized_railyard"]

    def run(self, ctx: Ctx) -> Ctx:
        ry_opt = self._maybe_call(
            "pipeline.core.optimization.railyard_optimizer", "optimize", ctx["railyard_layout"], ctx["optimized_timetable"]
        )
        ctx["optimized_railyard"] = ry_opt
        return ctx


class WriteCache(Step):
    name = "Write CACHE"
    uses = ["run_id"]
    adds = ["cache_path"]

    CACHE_DIR = Path(__file__).resolve().parent / ".." / "artifacts" / "_cache"

    def run(self, ctx: Ctx) -> Ctx:
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = self.CACHE_DIR / f"{ctx['run_id']}.pkl"
        with cache_path.open("wb") as fh:
            pickle.dump(dict(ctx), fh)
        ctx["cache_path"] = cache_path
        logger.info("    ↪ cache written → %s", cache_path.as_posix())
        return ctx


class WriteDatabase(Step):
    name = "Write Database"
    uses = ["optimized_railyard", "optimized_timetable"]
    adds = ["db_record_id"]

    DB_PATH = Path(__file__).resolve().parent / ".." / "database" / "runs.jsonl"

    def run(self, ctx: Ctx) -> Ctx:
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": ctx["run_id"],
            "country": ctx["country"],
            "start_ts": ctx["start_ts"],
            "optimized_railyard": bool(ctx.get("optimized_railyard")),
            "optimized_timetable": bool(ctx.get("optimized_timetable")),
            "finished_ts": datetime.utcnow().isoformat(),
        }
        with self.DB_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        ctx["db_record_id"] = ctx["run_id"]
        logger.info("    ↪ DB record appended → %s", self.DB_PATH.as_posix())
        return ctx


class EpochManager(Step):
    """
    Checks if an 'epoch' (i.e., grouping key like country + mode) already exists.
    If yes → return ACCEPT / REJECT decision in ctx["epoch_decision"].
    """

    name = "Epoch Manager"
    uses = ["country", "mode"]
    adds = ["epoch_decision"]

    EPOCH_PATH = Path(__file__).resolve().parent / ".." / "database" / "epochs.json"

    def run(self, ctx: Ctx) -> Ctx:
        key = f"{ctx['country'].lower()}:{ctx['mode']}"
        epochs = {}
        if self.EPOCH_PATH.exists():
            epochs = json.loads(self.EPOCH_PATH.read_text())
        if key in epochs:
            ctx["epoch_decision"] = "update"  # or maybe 'reject'
        else:
            epochs[key] = ctx["run_id"]
            self.EPOCH_PATH.write_text(json.dumps(epochs, indent=2))
            ctx["epoch_decision"] = "create"
        logger.info("    ↪ epoch decision: %s", ctx["epoch_decision"])
        return ctx


# -----------------------------------------------------------------------------
# Pipeline factory
# -----------------------------------------------------------------------------
def build_pipeline(mode: str) -> List[Step]:
    """
    mode == "learn" → still goes through same DAG; you can branch inside steps if needed.
    """
    return [
        Initialize(),
        ProcessingInput(),
        ProcessingTerrain(),
        ReferencingSimilarCountry(),
        ProcessingDemand(),
        ConstraintAnalyzer(),
        PlottingRoute(),
        NIMBYAnalyzer(),
        RouteOptimizer(),
        ChoosingTrainForRoute(),
        CreatingTimeTable(),
        OptimizingTimeTable(),
        RailyardPlotter(),
        RailyardOptimizer(),
        WriteCache(),
        WriteDatabase(),
        EpochManager(),
    ]


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_pipeline(
    country: str,
    mode: str = "learn",
    budget: Optional[float] = None,
    reference: Optional[str] = None,
    refresh: bool = False,
) -> Ctx:

    """
    High-level orchestration entry point—import & call this from tests too.
    Returns the final context for inspection.
    """
    ctx = Ctx(country=country, mode=mode, budget=budget, reference_country=reference, refresh=refresh)


    for step in build_pipeline(mode):
        ctx = step(ctx)

    logger.info("⏹  Pipeline finished – status dump ↓\n%s", ctx.dump())
    return ctx


# -----------------------------------------------------------------------------
# Typer CLI
# -----------------------------------------------------------------------------
cli = typer.Typer(add_completion=False)


@cli.command()
def run(
    country: str = typer.Option(..., help="Target country ISO name, e.g. Belgium"),
    mode: str = typer.Option("learn", help="Pipeline mode: learn | infer"),
    budget: Optional[float] = typer.Option(None, help="Budget cap for inference mode"),
    ref: Optional[str] = typer.Option(None, help="Use artifacts from this reference country"),
    refresh: bool = typer.Option(False, "--refresh", help="Force re-download of terrain, demand, etc."),
):
    """
    Run the pipeline end-to-end.  All flags map straight into `run_pipeline`.
    """
    run_pipeline(country=country, mode=mode, budget=budget, reference=ref, refresh=refresh)


# Allow python -m pipeline.pipeline_initiator ...
def main():
    cli()


if __name__ == "__main__":
    main()
