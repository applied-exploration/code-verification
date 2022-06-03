import torch as t
import faiss
from scipy import spatial


vectors_orig = t.load("data/temporary/features_2.pt")[0][0].numpy()
print(1)
index = faiss.index_factory(vectors_orig.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
vectors = faiss.normalize_L2(vectors_orig)
index.add(vectors)

q = vectors_orig[:1]
distance, index = index.search(faiss.normalize_L2(vectors_orig[:3]), 5)

print(distance, index)
# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

ncentroids = 100
niter = 20
verbose = True
kmeans = faiss.Kmeans(vectors_orig.shape[1], ncentroids, niter=niter, verbose=verbose)
kmeans.train(vectors)
