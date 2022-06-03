import torch as t
from annoy import AnnoyIndex

vectors = t.load("data/temporary/features_2.pt")[0][0].numpy()
print(1)
db = AnnoyIndex(vectors.shape[1], "angular")
for i, vector in enumerate(vectors):
    db.add_item(i, vector)
db.build(10)  # 10 trees


result = db.get_nns_by_vector(vectors[1], 10, include_distances=True)
print(result)
