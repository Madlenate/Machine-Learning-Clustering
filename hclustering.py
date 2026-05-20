
import sys
from clusterHelper import loadDataset, euclidean, printReport
import numpy as np
import json


def compute_distance_matrix(X):
  """
  Returns an (n x n) symmetric matrix of pairwise Euclidean distances.
  """
  n = len(X)
  D = np.zeros((n, n))

  for i in range(n):
    for j in range(i + 1, n):
      d = euclidean(X[i], X[j])
      D[i, j] = d
      D[j, i] = d

  return D


def cluster_distance(a_indices, b_indices, D, method="single"):
  """
  Computes the distance between two clusters using the distance matrix D.
  """
  distances = [D[i, j] for i in a_indices for j in b_indices]

  if method == "single":
    return min(distances)
  elif method == "complete":
    return max(distances)
  elif method == "average":
    return sum(distances) / len(distances)
  else:
    raise ValueError(f"Unknown linkage method: '{method}'")


def make_leaf(point, index):
  """
  Create a leaf node.
  The index is needed so we know which original data point gets assigned later.
  """
  return {
      "type": "leaf",
      "index": int(index),
      "data": list(point)
  }


def makeNode(height, left, right):
  return {
      "type": "node",
      "height": round(float(height), 6),
      "nodes": [left, right]
  }


def hclustering(X, D, linkage="single"):
  n = len(X)

  nodes = {i: make_leaf(X[i], i) for i in range(n)}
  indices = {i: [i] for i in range(n)}
  active = list(range(n))
  nextId = n

  while len(active) > 1:
    bestDist = float("inf")
    bestA, bestB = None, None

    for i in range(len(active)):
      for j in range(i + 1, len(active)):
        ca, cb = active[i], active[j]
        d = cluster_distance(indices[ca], indices[cb], D, linkage)

        if d < bestDist:
          bestDist = d
          bestA, bestB = ca, cb

    newNode = makeNode(bestDist, nodes[bestA], nodes[bestB])
    newIndices = indices[bestA] + indices[bestB]

    nodes[nextId] = newNode
    indices[nextId] = newIndices

    active.remove(bestA)
    active.remove(bestB)
    active.append(nextId)

    nextId += 1

  root = nodes[active[0]]
  root["type"] = "root"

  return root


def collectLeaves(node):
  """
  Collects the original indexes of all leaf nodes under this node.
  """
  if node["type"] == "leaf":
    return [node["index"]]

  leaves = []

  for child in node["nodes"]:
    leaves.extend(collectLeaves(child))

  return leaves


def cutTree(node, threshold, labels, clusterID):
  """
  Cuts the dendrogram at the given threshold and assigns labels.

  If a node's height is <= threshold, everything under that node becomes
  one cluster.
  """
  if node["type"] == "leaf":
    labels[node["index"]] = clusterID
    return clusterID + 1

  if node["height"] <= threshold:
    leafIndexes = collectLeaves(node)

    for idx in leafIndexes:
      labels[idx] = clusterID

    return clusterID + 1

  for child in node["nodes"]:
    clusterID = cutTree(child, threshold, labels, clusterID)

  return clusterID


def labelsToClusters(X, labels):
  """
  Converts labels array into list of clusters, where each cluster is a list of points.
  """
  clusters = []

  for label in sorted(set(labels)):
    pts = X[labels == label]
    clusters.append([list(p) for p in pts])

  return clusters


# PRINT HELPERS
def printDendrogram(root):
  dendroJson = json.dumps(root, indent=2)

  print("\nDendrogram")
  print(dendroJson)

  with open("dendrogram.json", "w") as f:
    f.write(dendroJson)

  print("\nDendrogram saved to dendrogram.json")


def printClusters(clusters, threshold, n):
  print(f"\n--Clusters at threshold {threshold}--")
  print(f"# of clusters: {len(clusters)}")
  print(f"# of data points: {n}")

  for i, cluster in enumerate(clusters):
    pts = [np.array(p) for p in cluster]
    centroid = np.mean(pts, axis=0).tolist()
    pct = len(cluster) / n * 100
    cStr = ", ".join(f"{v:.4f}" for v in centroid)

    print(f"\nCluster {i}: {len(cluster)} pts ({pct:.2f}%) centroid=[{cStr}]")

    MAX_SHOW = 19

    for pt in cluster[:MAX_SHOW]:
      print(" ", [round(float(v), 4) for v in pt])

    if len(cluster) > MAX_SHOW:
      print(f"    ... and {len(cluster) - MAX_SHOW} more")


def main():
  if len(sys.argv) < 2:
    print("Usage: python3 hclustering.py <dataset> [threshold] [--linkage single|complete|average]")
    return

  f = sys.argv[1]
  threshold = None
  linkage = "single"

  i = 2

  while i < len(sys.argv):
    if sys.argv[i] == "--linkage" and i + 1 < len(sys.argv):
      linkage = sys.argv[i + 1]
      i += 2
    else:
      try:
        threshold = float(sys.argv[i])
      except ValueError:
        print(f"Ignoring argument: {sys.argv[i]}")
      i += 1

  X, ignored, restrictions = loadDataset(f)

  print(f"Loaded {len(X)} points from '{f}'")
  print(f"Linkage: {linkage}")

  D = compute_distance_matrix(X)
  root = hclustering(X, D, linkage)

  printDendrogram(root)

  if threshold is not None:
    labels = np.full(len(X), -1)

    cutTree(root, threshold, labels, 0)

    print("\nUnassigned points:", np.sum(labels == -1))

    clusters = labelsToClusters(X, labels)

    groundTruth = None
    if "iris" in f.lower() and ignored is not None and ignored.shape[1] == 1:
      groundTruth = ignored

    printClusters(clusters, threshold, len(X))
    printReport(X, labels, groundTruth)


if __name__ == "__main__":
  main()
