# … existing imports, paths, DEFAULT_COSTS, etc. stay the same …

def parse_master_csv(path: pl.Path = INPUT) -> dict[str, pd.DataFrame]:
    df   = pd.read_csv(path)
    data: dict[str, pd.DataFrame] = {}

    # ── node-level  -----------------------------------------------------
    data["Stations"] = (
        df[["station_id", "city", "name", "n_tracks",
            "size_m2", "amenities", "overhead_wires"]]
        .dropna(subset=["station_id"])
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )

    data["Population-per-city"] = (
        df[["city", "population"]]
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    # lat/lon for every city present in the CSV
    data["City-coords"] = (
        df[["city", "lat", "lon"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    # ── edge-level  -----------------------------------------------------
    edge_rows = df[df["segment_id"].notna()]

    data["Tracks"] = (
        edge_rows[["segment_id", "city_a", "city_b", "distance_km",
                   "terrain_class", "allowed_train_types"]]
        .drop_duplicates("segment_id")
        .reset_index(drop=True)
    )

    data["Frequency"] = (
        edge_rows[["segment_id", "peak_trains_per_hour",
                   "offpeak_trains_per_hour"]]
        .drop_duplicates("segment_id")
        .reset_index(drop=True)
    )

    # ── project-wide  ---------------------------------------------------
    data["Budget"] = (
        df[["fiscal_year", "currency", "capex_million", "opex_million"]]
        .drop_duplicates("fiscal_year")
        .reset_index(drop=True)
    )

    # ── validation: every city in Tracks must have coordinates ----------
    track_cities = set(data["Tracks"]["city_a"]) | set(data["Tracks"]["city_b"])
    coord_cities = set(data["City-coords"]["city"])
    missing = track_cities - coord_cities
    if missing:
        raise ValueError(
            f"🚫  Missing lat/lon for: {', '.join(sorted(missing))}. "
            "Add these rows to master-data.csv."
        )

    return data
