"""
Calculate station footprint (platforms & area) from timetable.
"""
import pandas as pd
from collections import Counter

def station_specs(timetable: pd.DataFrame,
                  dwell_min: int = 4,
                  platform_length_m: int = 250
                 ) -> pd.DataFrame:
    """
    Returns DataFrame: city, peak_trains_per_hour, n_platforms, gross_m2
    """
    # count departures per city per hour
    timetable["hour"] = pd.to_datetime(timetable["dep_time"]).dt.hour
    counts = timetable.groupby(["dep_city", "hour"]).size()
    peak = counts.groupby("dep_city").max().rename("trains_ph")

    df = peak.to_frame().reset_index()
    df["n_platforms"] = (df["trains_ph"] * dwell_min / 60).apply(lambda x: int(x+1))
    df["gross_m2"]    = df["n_platforms"] * platform_length_m * 10   # 10 m width
    return df
