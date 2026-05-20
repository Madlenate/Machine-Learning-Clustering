
import os
import re
import pandas as pd

def getMetrics(pattern, text):
  match = re.search(pattern, text)
  if match:
    return match.group(1)
  return None


def fSummary(fPath):
  fileName = os.path.basename(fPath)
  with open(fPath, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

  silhouette = getMetrics(r"Overall Silhouette Score: \s*([-+]?\d*\.\d+|\d+)", text)
  rand = getMetrics(r"Rand Index: \s*([-+]?\d*\.\d+|\d+)", text)
  ratio = getMetrics(r"Radius / Intercluster Distance Ratio: \s*([-+]?\d*\.\d+|\d+)", text)
  dataPts = getMetrics(r"# of data points: \s*(\d+)", text)
  outliers = getMetrics(r"# of outliers: \s*(\d+)", text)
  clusterLabels = getMetrics(r"Cluster labels found: \s*(\[.*?\])", text)

  return {"file": fileName,
          "silhouette": float(silhouette) if silhouette else None,
          "rand_index": float(rand) if rand else None,
          "radius_intercluster_ratio": float(ratio) if ratio else None,
          "dataPts": int(dataPts) if dataPts else None,
          "outliers": int(outliers) if outliers else None,
          "clusterLabels": clusterLabels}


def main():
  rows = []

  for fileName in os.listdir("outputs"):
    if fileName.endswith(".txt"):
      fPath = os.path.join("outputs", fileName)
      rows.append(fSummary(fPath))

  df = pd.DataFrame(rows)
  df = df.sort_values(by="file")
  df.to_csv("results.csv", index=False)
  print(df)
  print("\nSaved to results.csv")

if __name__ == "__main__":
  main()
