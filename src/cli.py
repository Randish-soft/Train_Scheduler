# ── src/cli.py ──────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from . import INPUT_DIR
from .io import load_scenario
from .pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
LOG = logging.getLogger("bcpc.cli")


# ───────────────────────── helper: list CSVs ───────────────────────────────
def _find_csv_files() -> list[Path]:
    return sorted(INPUT_DIR.glob("*.csv"))


# ─────────────────── interactive arrow-key picker ──────────────────────────
def _arrow_menu(options: Sequence[str]) -> int:
    """Return index chosen via ↑/↓ + Enter (uses curses)."""
    import curses

    def _menu(stdscr):
        curses.curs_set(0)
        idx = 0
        while True:
            stdscr.clear()
            for i, opt in enumerate(options):
                mode = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addstr(i, 0, opt, mode)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(options)
            elif key in (curses.KEY_ENTER, 10, 13):
                return idx

    return curses.wrapper(_menu)


def _select_file(paths: Sequence[Path]) -> Path:
    opts = [p.name for p in paths]
    try:
        choice = _arrow_menu(opts)
        return paths[choice]
    except Exception:  # fallback if curses isn't available (e.g. Windows)
        print("\n".join(f"[{i}] {p.name}" for i, p in enumerate(paths)))
        while True:
            sel = input("Select file number: ")
            if sel.isdigit() and int(sel) in range(len(paths)):
                return paths[int(sel)]
            print("Invalid selection.")


# ──────────────────────────── main entry-point ─────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCPC scenario pipeline")
    parser.add_argument("--csv", help="Path to scenario CSV")
    args = parser.parse_args()

    # ── locate CSV ──
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csvs = _find_csv_files()
        if not csvs:
            LOG.error("No CSV files found in %s", INPUT_DIR)
            sys.exit(1)
        if len(csvs) == 1:
            csv_path = csvs[0]
            LOG.info("Found single CSV: %s", csv_path.name)
        else:
            LOG.info("Multiple CSVs detected – pick one with ↑ ↓ and Enter")
            csv_path = _select_file(csvs)

    # ── load & run ──
    rows = load_scenario(csv_path)            # list[ScenarioRow]
    LOG.info("Using CSV %s (%d rows)", csv_path.name, len(rows))

    for row in rows:
        run_pipeline(row, rows)               # <-- pass ALL rows


if __name__ == "__main__":
    main()
