
import sys
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, rand_score
from clusterHelper import loadDataset

bestParameters = {
  "iris.csv": {
    "kmeans_k": 3,
    "dbscan_eps": 0.7,
    "dbscan_min_samples": 4,
    "h_threshold": 1.5,
    "h_linkage": "single"
  },
  "4clusters.csv": {
    "kmeans_k": 4,
    "dbscan_eps": 5.0,
    "dbscan_min_samples": 4,
    "h_threshold": 10.0,
    "h_linkage": "single"
  },
  "mammal_milk.csv": {
    "kmeans_k": 2,
    "dbscan_eps": 5.0,
    "dbscan_min_samples": 3,
    "h_threshold": 10.0,
    "h_linkage": "single"
  },
  "planets.csv": {
    "kmeans_k": 2,
    "dbscan_eps": 10.0,
    "dbscan_min_samples": 3,
    "h_threshold": 100.0,
    "h_linkage": "single"
  },
  "AccidentsSet03.csv": {
    "kmeans_k": 3,
    "dbscan_eps": 1.0,
    "dbscan_min_samples": 3,
    "h_threshold": 2.0,
    "h_linkage": "single"}}

def safeSilhouette(X, labels):
  unique = set(labels)

  if len(unique) < 2:
    return None
  if len(unique) >= len(X):
    return None
  try:
    return silhouette_score(X, labels)
  except Exception:
    return None

def getGroundTruth(fileName, ignored):
  # only Iris has true class labels in the ignored column.
  # other ignored columns are names/IDs, not ground truth.
  if "iris" in fileName.lower() and ignored is not None and ignored.shape[1] == 1:
    return ignored.flatten()
  return None

def printSklearnReport(X, labels, groundTruth, alg, paramsText):
  labels = np.array(labels)
  unique = sorted(set(labels))

  print(f"\nAlgorithm: {alg}")
  print(f"Parameters: {paramsText}")
  print("\n--Overall Cluster Report--")
  print(f"# of data points: {len(X)}")
  print(f"Cluster labels found: {[int(label) for label in unique]}")

  if -1 in unique:
    outliers = np.sum(labels == -1)
    pct = outliers / len(labels) * 100
    print(f"# of outliers: {outliers}")
    print(f"Percent outliers: {pct:.2f}%")

  sil = safeSilhouette(X, labels)

  if sil is None:
    print("Overall Silhouette Score: NaN")
  else:
    print(f"Overall Silhouette Score: {sil}")

  if groundTruth is not None:
    ri = rand_score(groundTruth, labels)
    print(f"Rand Index: {ri}")

  print("\n--Cluster Sizes--")
  for label in unique:
    count = np.sum(labels == label)

    if label == -1:
      print(f"Noise / Outliers: {count}")
    else:
      print(f"Cluster {int(label)}: {count}")

def compareDataset(fileName):
  basename = os.path.basename(fileName)

  if basename not in bestParameters:
    print(f"No saved parameters for {basename}. Using default values.")
    params = {
      "kmeans_k": 3,
      "dbscan_eps": 1.0,
      "dbscan_min_samples": 3,
      "h_threshold": 2.0,
      "h_linkage": "single"}
  else:
    params = bestParameters[basename]

  X, ignored, restrictions = loadDataset(fileName)
  groundTruth = getGroundTruth(basename, ignored)

  print(f"\n--SKLEARN COMPARISON FOR: {basename}--")

  # sklearn KMeans
  kmeansModel = KMeans(n_clusters=params["kmeans_k"], random_state=0, n_init=10)
  kmeansLabels = kmeansModel.fit_predict(X)
  printSklearnReport(X, kmeansLabels, groundTruth, "sklearn KMeans", f"k = {params['kmeans_k']}")

  # sklearn DBSCAN
  dbscanModel = DBSCAN(eps=params["dbscan_eps"], min_samples=params["dbscan_min_samples"])
  dbscanLabels = dbscanModel.fit_predict(X)
  printSklearnReport(X, dbscanLabels, groundTruth, "sklearn DBSCAN", f"eps = {params['dbscan_eps']}, min_samples = {params['dbscan_min_samples']}")

  # sklearn Agglomerative Clustering using distance threshold
  hModel = AgglomerativeClustering(n_clusters=None, distance_threshold=params["h_threshold"], linkage=params["h_linkage"])
  hLabels = hModel.fit_predict(X)
  printSklearnReport(X, hLabels, groundTruth, "sklearn AgglomerativeClustering", f"distance_threshold = {params['h_threshold']}, linkage = {params['h_linkage']}")

def main():
  if len(sys.argv) >= 2:
    compareDataset(sys.argv[1])
  else:
    datasets = [
      "iris.csv",
      "4clusters.csv",
      "mammal_milk.csv",
      "planets.csv",
      "AccidentsSet03.csv"]

    for dataset in datasets:
      if os.path.exists(dataset):
        compareDataset(dataset)
      else:
        print(f"Skipping {dataset}: file not found.")

if __name__ == "__main__":
  main()
