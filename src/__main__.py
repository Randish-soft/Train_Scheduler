# ── src/__main__.py ──────────────────────────────────────────
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


def _find_csv_files() -> list[Path]:
    return sorted(INPUT_DIR.glob("*.csv"))


# ---------- interactive selection helpers ---------- #
def _arrow_menu(options: Sequence[str]) -> int:
    """Return index chosen via ↑/↓ + Enter (curses)."""
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
    except Exception:  # noqa: BLE001
        # Fallback: numeric prompt
        print("\n".join(f"[{i}] {p.name}" for i, p in enumerate(paths)))
        while True:
            sel = input("Select file number: ")
            if sel.isdigit() and int(sel) in range(len(paths)):
                return paths[int(sel)]
            print("Invalid selection.")


# ---------- main ---------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCPC scenario pipeline")
    parser.add_argument("--csv", help="Path to scenario CSV")
    args = parser.parse_args()

    csv_path: Path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_files = _find_csv_files()
        if not csv_files:
            LOG.error("No CSV files found in %s", INPUT_DIR)
            sys.exit(1)
        if len(csv_files) == 1:
            csv_path = csv_files[0]
            LOG.info("Found single CSV: %s", csv_path.name)
        else:
            LOG.info("Multiple CSVs detected – pick one with ↑ ↓ and Enter")
            csv_path = _select_file(csv_files)

    LOG.info("Using CSV %s", csv_path)
    for row in load_scenario(csv_path):
        run_pipeline(row)


if __name__ == "__main__":
    main()
# ───────────────────────────────────────────────────────────
