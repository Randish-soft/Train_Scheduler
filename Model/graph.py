import networkx as nx
from Model.dataload import read_json
from Model.geom     import haversine

def build_graph() -> nx.Graph:
    tracks   = read_json("Tracks")
    coords   = read_json("City-coords").set_index("city")

    G = nx.Graph()
    for _, row in tracks.iterrows():
        u, v = row["city_a"], row["city_b"]

        # Look-ups will always succeed because dataload.py already validated
        lat_u, lon_u = coords.loc[u, ["lat", "lon"]]
        lat_v, lon_v = coords.loc[v, ["lat", "lon"]]

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
