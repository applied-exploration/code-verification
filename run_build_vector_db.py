import torch as t
from annoy import AnnoyIndex
from typing import Tuple
import pandas as pd
from config import preprocess_config

vectors = t.load("data/derived/embeddings_flat.pt").numpy()
print(1)
db = AnnoyIndex(vectors.shape[1], "angular")
for i, vector in enumerate(vectors):
    db.add_item(i, vector)
db.build(100)  # 10 trees


result = db.get_nns_by_vector(vectors[1], 10, include_distances=True)
print(result)

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=100)
kmeans.fit(vectors)
centroids = kmeans.cluster_centers_

original_data = pd.read_json(f"data/derived/{preprocess_config.dataset}.json")


for centroid in centroids:
    result = db.get_nns_by_vector(centroid, 1, include_distances=False)
    print(original_data.iloc[result[0]])
