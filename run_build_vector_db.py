import torch as t
from annoy import AnnoyIndex

vectors = t.load("data/temporary/features_2.pt")[0][0].numpy()
print(1)
t = AnnoyIndex(vectors.shape[1], 'angular')
t.add_item()
index.add(vectors)  # add vectors to the index
t.build(10) # 10 trees


k = 4  # we want to see 4 nearest neighbors
D, indicies = index.search(vectors[:1], k)  # sanity check
print(I)
print(D)
