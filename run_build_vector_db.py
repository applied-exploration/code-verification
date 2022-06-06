import torch as t
from annoy import AnnoyIndex
from typing import Tuple
import pandas as pd

vectors = t.load("data/derived/embeddings_flat.pt").numpy()
print(1)
db = AnnoyIndex(vectors.shape[1], "angular")
for i, vector in enumerate(vectors):
    db.add_item(i, vector)
db.build(10)  # 10 trees


result = db.get_nns_by_vector(vectors[1], 10, include_distances=True)
print(result)

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=100)
kmeans.fit(vectors)
centroids = kmeans.cluster_centers_

id_map = t.load("data/derived/id_to_info_map.pt")
original_data = t.load("data/derived/embeddings_per_file.pt")

data = pd.read_json("data/derived/python-pytorch.json")


def lookup_in_original_data(lookup_id: Tuple[int, int]) -> dict:
    return original_data[lookup_id[0]][lookup_id[1]]


for centroid in centroids:
    result = db.get_nns_by_vector(centroid, 1, include_distances=False)
    lookup_in_original_data(id_map[result[0]])
