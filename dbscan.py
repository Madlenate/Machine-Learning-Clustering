
import sys
import numpy as np
from clusterHelper import loadDataset, printReport, euclidean

def getNeighbors(X, ptIdx, e):
    # returns a list of indexes of all points within e distance of ptIdx

    neighborsList = []

    for i in range(len(X)):
        dist = euclidean(X[ptIdx], X[i])
        if dist <= e:
            neighborsList.append(i)

    return neighborsList

def dbscan(X, e, minPts):
    # DBSCAN clustering algorithm.
    # -1 -> noise/outlier
    # 0, 1, 2, ... -> cluster labels

    labels = np.full(len(X), -1)
    visited = np.full(len(X), False)

    id = 0

    for ptIdx in range(len(X)):
        if visited[ptIdx]:
            continue
        visited[ptIdx] = True

        neighborsList = getNeighbors(X, ptIdx, e)

        if len(neighborsList) < minPts:
            labels[ptIdx] = -1
        else:
            expandCluster(X, labels, visited, ptIdx, neighborsList, id, e, minPts)
            id += 1

    return labels


def expandCluster(X, labels, visited, ptIdx, neighborsList, id, e, minPts):
    # from core point, expand cluster

    labels[ptIdx] = id

    i = 0
    while i < len(neighborsList):
        neighborIdx = neighborsList[i]

        if not visited[neighborIdx]:
            visited[neighborIdx] = True
            nNeighbors = getNeighbors(X, neighborIdx, e)

            if len(nNeighbors) >= minPts:
                for n in nNeighbors:
                    if n not in neighborsList:
                        neighborsList.append(n)
        if labels[neighborIdx] == -1:
            labels[neighborIdx] = id

        i += 1

def printOutlierSummary(labels):
    labels = np.array(labels)
    totalPts = len(labels)
    numOutliers = np.sum(labels == -1)
    percentOutliers = (numOutliers / totalPts) * 100
    print("--DBSCAN Outlier Summary--")
    print(f"# of outliers: {numOutliers}")
    print(f"Percent outliers: {percentOutliers:.2f}%")

def main():
    if len(sys.argv) < 4:
        print("python3 dbscan.py <f> <e> <minPts>")
        return

    f = sys.argv[1]
    e = float(sys.argv[2])
    minPts = int(sys.argv[3])

    X, ignored, restrictions = loadDataset(f)

    labels = dbscan(X, e, minPts)

    groundTruth = None
    if "iris" in f.lower() and ignored is not None and ignored.shape[1] == 1:
      groundTruth = ignored

    print(f"DBSCAN finished with epsilon = {e}, minPts = {minPts}")
    printOutlierSummary(labels)
    printReport(X, labels, groundTruth)


if __name__ == "__main__":
    main()
