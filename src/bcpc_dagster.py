from pathlib import Path
from dagster import In, Out, op, job, ScheduleDefinition, get_run_context
from src.scenario_io import load_scenario
from src.pipeline import run_pipeline

CSV_DEFAULT = "input/lebanon_cities_2024.csv"


@op(
    ins={"csv_path": In(str)},
    out={"geojson_path": Out(str)},
    description="BCPC pipeline run; artefacts stored per Dagster run-ID",
)
def bcpc_op(csv_path: str) -> str:
    ctx = get_run_context()
    run_dir = Path("output") / ctx.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_scenario(Path(csv_path))
    design, _breakdown, geojson = run_pipeline(rows[0])

    dest = run_dir / Path(geojson).name
    Path(geojson).rename(dest)

    (run_dir / "summary.txt").write_text(
        f"daily_ridership={design.ridership_daily}\n"
        f"capex_eur={design.cost_eur:,.0f}\n"
    )
    return str(dest)


bcpc_job = job(
    name="bcpc_job",
    config={
        "ops": {
            "bcpc_op": {"inputs": {"csv_path": CSV_DEFAULT}}
        }
    },
)

hourly_schedule = ScheduleDefinition(
    name="bcpc_hourly",
    cron_schedule="0 * * * *",   # every hour on the hour
    job=bcpc_job,
    execution_timezone="UTC",
)
