
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from clusterHelper import loadDataset
from kmeans import kmeans

bestKmeans = {
  "iris.csv": 3,
  "4clusters.csv": 4,
  "mammal_milk.csv": 2,
  "planets.csv": 2,
  "AccidentsSet03.csv": 3}

def makePlot(filename):
  X, ignored, restrictions = loadDataset(filename)
  basename = os.path.basename(filename)

  if basename not in bestKmeans:
    print(f"No k value saved for {basename}")
    return

  k = bestKmeans[basename]
  labels, centroids = kmeans(X, k)
  plt.figure(figsize=(7, 5))

  # first two clustering columns for visualization
  plt.scatter(X[:, 0], X[:, 1], c=labels)
  # plot centroids if they have 2+ dimensions
  if centroids.shape[1] >= 2:
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=120, linewidths=3)

  plt.title(f"K-means Clustering: {basename}, k={k}")
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  os.makedirs("figures", exist_ok=True)
  outname = basename.replace(".csv", "_kmeans.png")
  outpath = os.path.join("figures", outname)
  plt.savefig(outpath, bbox_inches="tight", dpi=200)
  plt.close()
  print(f"Saved {outpath}")

def main():
  if len(sys.argv) >= 2:
    makePlot(sys.argv[1])
  else:
    datasets = [
      "iris.csv",
      "4clusters.csv",
      "mammal_milk.csv",
      "planets.csv",
      "AccidentsSet03.csv"]

    for dataset in datasets:
      if os.path.exists(dataset):
        makePlot(dataset)
      else:
        print(f"Skipping {dataset}: file not found.")

if __name__ == "__main__":
  main()
