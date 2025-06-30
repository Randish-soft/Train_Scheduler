import networkx as nx
from .dataload import read_json

def build_graph() -> nx.Graph:
    tracks  = read_json("Tracks")
    terrain = read_json("Terrain").set_index("segment_id")["grade"]
    G = nx.Graph()
    for _, row in tracks.iterrows():
        u, v = row["city_a"], row["city_b"]
        seg   = row["segment_id"]
        G.add_edge(
            u, v,
            distance=row["distance_km"],
            grade=terrain[seg],
            train_types=row["allowed_trains"]
        )
    return G
