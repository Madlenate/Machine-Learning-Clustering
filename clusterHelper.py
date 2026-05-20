
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, rand_score

def loadDataset(f):
    # read first line manually to get the number of real columns
    with open(f, "r") as file:
        firstLine = file.readline().strip()

    restrictions = np.array([int(x.strip()) for x in firstLine.split(",") if x.strip() != ""])
    numCols = len(restrictions)

    # usecols ignores accidental trailing empty columns, like the extra comma in AccidentsSet03.csv
    df = pd.read_csv(f, header=None, usecols=range(numCols), skipinitialspace=True)

    data = df.iloc[1:].reset_index(drop=True)

    colsUsed = [i for i, val in enumerate(restrictions) if val == 1]
    colsIgnored = [i for i, val in enumerate(restrictions) if val == 0]
    X = data.iloc[:, colsUsed].astype(float).to_numpy()

    ignored = None
    if len(colsIgnored) > 0:
        ignored = data.iloc[:, colsIgnored].to_numpy()

    return X, ignored, restrictions

def euclidean(a, b):
  return np.sqrt(np.sum((a - b) ** 2))

def centroid(pts):
  return np.mean(pts, axis = 0)

def clusterRad(pts, centroid):
  # returns max dist from centroid to any cluster point
  if len(pts) == 0:
    return 0
  return max([euclidean(pt, centroid) for pt in pts])

def interclusterDist(c1, c2):
  return euclidean(c1, c2)

def radInterclusterRatio(X, labels):
  # average cluster radius / minimum distance between cluster centroids
  # want this number to be small
  unique = [label for label in sorted(set(labels)) if label != -1]

  if len(unique) < 2:
    return None

  centroids = {}
  radii = {}

  for label in unique:
    pts = X[np.array(labels) == label]
    c = centroid(pts)
    rad = clusterRad(pts, c)
    centroids[label] = c
    radii[label] = rad

  avgRad = np.mean(list(radii.values()))

  minInterclusterDist = float("inf")
  for i in range(len(unique)):
    for j in range(i + 1, len(unique)):
      dist = interclusterDist(centroids[unique[i]], centroids[unique[j]])
      if dist < minInterclusterDist:
        minInterclusterDist = dist

  if minInterclusterDist == 0:
    return None

  return avgRad / minInterclusterDist

def printReport(X, labels, groundTruth = None):
  # prints general metrics and cluster-by-cluster metrics
  labels = np.array(labels)
  unique = sorted(set(labels))

  print("--Overall Cluster Report")
  print(f"# of data points: {len(X)}")
  print(f"Cluster labels found: {[int(label) for label in unique]}")

  nonNoise = [label for label in unique if label != -1]
  if len(nonNoise) >= 2:
    ratio = radInterclusterRatio(X, labels)
    print(f"Radius / Intercluster Distance Ratio: {ratio}")

    try:
      silhouetteScore = silhouette_score(X, labels)
      print(f"Overall Silhouette Score: {silhouetteScore}")
    except Exception as e:
      print(f"Silhouette score could not be computed ({e})")
  else:
    print("Not enough clusters for silhouette score")
    print("Not enough clusters for radius / intercluster distance ratio")

  if groundTruth is not None:
    try:
      groundTruth = groundTruth.flatten()
      randIdx = rand_score(groundTruth, labels)
      print(f"Rand Index: {randIdx}")
    except Exception as e:
      print(f"Rand index could not be computed ({e})")

  print("\n--Cluster Details--")

  try:
    silSamples = silhouette_samples(X, labels)
  except Exception as e:
    silSamples = None

  for label in unique:
    pts = X[labels == label]

    if label == -1:
      print("\n--Outliers / Noise--")
    else:
      print(f"\n--Cluster {label}--")

    print(f"# of pts: {len(pts)}")
    if len(pts) > 0 and label != -1:
      c = centroid(pts)
      rad = clusterRad(pts, c)
      print(f"Centroid: {c}")
      print(f"Radius: {rad}")

      if silSamples is not None:
        silClusters = silSamples[labels == label]
        print(f"Cluster Silhouette Score: {np.mean(silClusters)}")

    if len(pts) <= 10:
      print("Points: ")
      for pt in pts:
        print(pt)
    else:
      print("Representative points: ")
      for pt in pts[:3]:
        print(pt)
