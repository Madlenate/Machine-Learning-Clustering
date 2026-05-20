
import sys
import numpy as np
from clusterHelper import loadDataset, euclidean, printReport

def initializeCentroids(X, k):
  # randomly choose k points and set them as the centroids

  idxs = np.random.choice(len(X), size = k, replace = False)
  centroids = X[idxs]

  return centroids

def ptsToClusters(X, centroids):
  # find nearest centroid for each point

  labels = []
  for pt in X:
    dist = [euclidean(pt, centroid) for centroid in centroids]
    labels.append(np.argmin(dist))

  return np.array(labels)

def recompCentroids(X, labels, k, prevCentroids):
  # recompute each centroid as the mean of all cluster points
  # keep previous centroid if cluster ends up being empty

  newCentroids = []

  for id in range(k):
    pts = X[labels == id]

    if len(pts) == 0:
      newCentroids.append(prevCentroids[id])
    else:
      newCentroids.append(np.mean(pts, axis = 0))

  return np.array(newCentroids)

def kmeans(X, k, maxIter = 100):
  # k-means clustering alg

  centroids = initializeCentroids(X, k)
  for iter in range(maxIter):
    labels = ptsToClusters(X, centroids)
    newCentroids = recompCentroids(X, labels, k, centroids)

    if np.allclose(centroids, newCentroids):
      print(f"k-means converged after {iter + 1} iterations")
      break

    centroids = newCentroids

  return labels, centroids

def main():
  if len(sys.argv) < 3:
    print("python3 kmeans.py <dataset> <k>")
    return

  f = sys.argv[1]
  k = int(sys.argv[2])

  X, ignored, restrictions = loadDataset(f)

  labels, centroids = kmeans(X, k)

  groundTruth = None
  if "iris" in f.lower() and ignored is not None and ignored.shape[1] == 1:
    groundTruth = ignored

  printReport(X, labels, groundTruth)

if __name__ == "__main__":
  main()
