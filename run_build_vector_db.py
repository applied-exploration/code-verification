from pprint import pprint
import torch as t
from annoy import AnnoyIndex
import pandas as pd
from tqdm import tqdm
from os.path import exists
from sklearn.cluster import MiniBatchKMeans

force_rebuild_index = False
index_file = "data/derived/annoy_index.ann"
vectors = t.load("data/derived/embeddings_flat.pt").numpy()

if not exists("data/derived/annoy_index.ann"):
    db = AnnoyIndex(vectors.shape[1], "angular")
    print("Add vectors")

    for i, vector in tqdm(enumerate(vectors)):
        db.add_item(i, vector)

    print("Build index")
    db.build(100)
    db.save(index_file)
else:
    print("Load index")
    db = AnnoyIndex(vectors.shape[1], "angular")
    db.load(index_file)


print("Search for vector")

result = db.get_nns_by_vector(vectors[1], 10, include_distances=True)
print(result)

print("Start clustering")

kmeans = MiniBatchKMeans(n_clusters=300)
kmeans.fit(vectors)
centroids = kmeans.cluster_centers_

original_data = pd.read_json(
    "data/original/TSSB-3M/file-0.jsonl.gz", lines=True, compression="gzip"
)


examples_per_cluster = []
for centroid in centroids[:5]:
    result = db.get_nns_by_vector(centroid, 5, include_distances=False)
    examples_per_cluster.append(original_data.iloc[result]["diff"].to_list())


pprint(examples_per_cluster)
