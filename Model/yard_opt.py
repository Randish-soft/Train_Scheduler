"""
Pick up to `max_yards` yard cities (stub = first N distinct yard rows).
Replace with p-median formulation later.
"""
import pandas as pd
from Model.dataload import read_json


def optimise_yards(timetable: pd.DataFrame,
                   max_yards: int = 4
                  ) -> pd.DataFrame:
    yards_df = read_json("Railyard-position")
    chosen = yards_df.head(max_yards).copy()
    chosen["chosen"] = True
    return chosen
