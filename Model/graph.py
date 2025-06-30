import networkx as nx
from Model.dataload import read_json
from Model.geom     import haversine

def build_graph() -> nx.Graph:
    tracks   = read_json("Tracks")
    coords   = read_json("City-coords").set_index("city")[["lat", "lon"]]

    G = nx.Graph()
    for _, row in tracks.iterrows():
        u, v = row["city_a"], row["city_b"]
        try:
            lat_u, lon_u = coords.loc[u]
            lat_v, lon_v = coords.loc[v]
        except KeyError as e:
            raise ValueError(f"Missing lat/lon for city {e.args[0]} — "
                             "check City-coords.json") from None

        geom_km = haversine(lat_u, lon_u, lat_v, lon_v)
        G.add_edge(
            u, v,
            segment_id = row["segment_id"],
            geom_km    = geom_km,
            design_km  = row["distance_km"],
            terrain    = row["terrain_class"],
            train_types= row["allowed_train_types"]
        )
    return G
