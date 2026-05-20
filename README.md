CSC 466 Lab 4
Clustering 
By Nathan Madlansacay & Winnie Trinh
Study Design
This lab implements and evaluates three unsupervised clustering algorithms K-Means, Agglomerative Hierarchical Clustering, and DBSCAN  applied across five datasets of varying structure and dimensionality. All three implementations share a common data-loading and evaluation framework defined in clusterHelper.py.
The helper module (clusterHelper.py) provides:
•       loadDataset: reads a CSV file where the first row is a binary restriction mask (1 = include feature, 0 = ignore). Ignored columns are retained separately as potential ground-truth labels.
•       euclidean: standard Euclidean distance, used as the distance measure across all three algorithms.
•       printReport: computes and displays overall silhouette score, per-cluster silhouette, Rand index (when ground truth is available), and the radius/intercluster-distance ratio. For Iris, the ignored column (species name) is used as ground truth.
•       radInterclusterRatio: computes average cluster radius divided by the minimum centroid-to-centroid distance. A smaller ratio indicates better-separated, more compact clusters.
K-means Clustering
Initialization: k centroids are chosen uniformly at random from the data (random seeding, no K-Means++ heuristic).
Assignment: Each point is assigned to its nearest centroid by Euclidean distance.
Update: Centroids are recomputed as the mean of all assigned points. If a cluster becomes empty, its previous centroid is retained.
Convergence: Iteration halts when centroids do not change (np.allclose) or after a maximum of 100 iterations.
Parameters: k (number of clusters). Experimented with k = 2, 3, 4, 5 depending on dataset.
Because random initialization can produce different local optima, multiple runs were compared and the result with the highest silhouette score was reported.
Hierarchical Clustering
Distance matrix: An n×n symmetric pairwise Euclidean distance matrix is computed upfront.
Linkage: Three linkage methods are supported: single-link (minimum distance between cluster members), complete-link (maximum), and average-link (mean). All runs in this study used single linkage unless otherwise noted.
Merging: Starting with n singleton clusters, the two closest clusters are greedily merged at each step. The resulting dendrogram is output as a JSON tree.
Cutting: A distance threshold is applied to the dendrogram: subtrees whose root height is at or below the threshold are collapsed into a single cluster. The threshold directly controls the number of resulting clusters.
Parameters: distance threshold (explored across multiple values per dataset), linkage method.
DBSCAN Clustering 
Core point criterion: A point is a core point if its ε-neighborhood contains at least minPts points (including itself).
Cluster expansion: A BFS/DFS expansion from each unvisited core point propagates the cluster label to all density-reachable points.
Noise: Points that are not core points and not reachable from any core point are labeled −1 (noise/outlier).
Parameters: ε (epsilon radius) and minPts. These were tuned by grid search: ε ranged from 0.5–15.0 and minPts from 3–5 depending on dataset scale.








Results
Iris Data Set
Metric
Value
Silhouette Score
0.5526
Rand Index
0.8797
Radius/IC Ratio
0.8216
Cluster Sizes
50 / 50 / 50 (approx.)

K-Means  —  k = 3  [Appendix A-1]
python kmeans.py iris.csv 3  ->  outputs/kmeans_iris_k3.txt
K-Means at k=3 is the best K-Means result for Iris and matches the known number of species. The three clusters broadly correspond to setosa, versicolor, and virginica, with the primary difficulty being the partial overlap between versicolor and virginica in feature space.
K-Means at k=4 reduced the silhouette score to 0.4171 by splitting versicolor/virginica into artificial sub-clusters.
Metric
Value
Silhouette Score
0.5016
Rand Index
0.7771 (2-cluster partition vs. 3-class truth)
Noise Points
3 (2.00%)
Cluster Sizes
Cluster 0: 50 | Cluster 1: 97


DBSCAN  —  ε = 0.7, minPts = 4  [Appendix A-2]
python dbscan.py iris.csv 0.7 4  ->  outputs/dbscan_iris_eps07_min4.txt
At ε=0.7, DBSCAN found two core clusters plus 3 noise points (2% of data). Smaller ε (0.5) produced 13 noise points and a lower silhouette of 0.381. Larger ε would collapse versicolor and virginica. The two main clusters found (setosa vs. versicolor+virginica combined) reflect the density gap between setosa and the others.



Metric
Value
Silhouette Score
0.6864
Rand Index
0.7763 (2 vs. 3 true classes)
Cluster Sizes
Cluster 0: 100 | Cluster 1: 50
Observation
Single-link chaining: versicolor + virginica merged


Hierarchical Clustering  —  threshold = 1.5, single linkage  [Appendix A-3]
python hclustering.py iris.csv 1.5  ->  outputs/hcluster_iris_t15.txt
Single-linkage hierarchical clustering at threshold 1.5 produced only 2 clusters (sizes 100 and 50), achieving the highest silhouette score of any Iris run at 0.6864. This is because single linkage chained versicolor and virginica together under one cluster, leaving setosa as the second cluster. While this looks good by silhouette, the Rand index (0.7763) reveals it does not align well with the three-class ground truth. Higher thresholds (2.0, 2.5) merged everything into one cluster.
4clusters Dataset
K-Means  —  k = 3  [Appendix B-1]
python kmeans.py 4clusters.csv 3  →  outputs/kmeans_4clusters_k3.txt
Counterintuitively, k=3 yielded a higher silhouette score (0.6710) than k=4 (0.5699) on this dataset. Inspection suggests that two of the four blobs are close enough that merging them into one cluster increases average silhouette. k=4 did find all four groups but with lower cohesion scores per cluster.

Metric
Value
Silhouette Score (k=3)
0.6710
Silhouette Score (k=4)
0.5699
Clusters (k=3)
[0, 1, 2]
Radius/IC Ratio
0.3816



Metric
Value
Silhouette Score
0.5432
Outliers
6 (15.38%)
Clusters Found
4 + noise
Radius/IC Ratio
0.4680

DBSCAN  —  ε = 5.0, minPts = 4  [Appendix B-2]
python dbscan.py 4clusters.csv 5.0 4  →  outputs/dbscan_4clusters_eps50_min4.txt
DBSCAN at ε=5.0 correctly found 4 clusters but labeled 6 of 39 points (15.4%) as noise. Increasing ε to 8.0–10.0 reduced noise to 0 but collapsed two clusters into one, again yielding 3 effective clusters. The silhouette score at ε=8.0+ equaled the hierarchical/k=3 result (0.6710), since the resulting partition is identical.
Metric
Value
Our Silhouette Score
0.5460 (8 clusters)
Sklearn Silhouette Score
0.7923 (2 clusters)
Discrepancy
Same threshold, same linkage — different # clusters


Hierarchical Clustering  —  threshold = 10.0, single linkage  [Appendix B-3]
python hclustering.py 4clusters.csv 10.0  →  outputs/hcluster_4clusters_t10.txt
At threshold 10.0, the dendrogram cut produces 3 clusters (matching the k=3 result) with silhouette 0.6710. At threshold 5.0, over-fragmentation yields 7 micro-clusters with silhouette 0.384. Thresholds above 15.0 merge everything into one cluster.
Metric
Value
Silhouette Score
0.6710
Clusters
[0, 1, 2]
Radius/IC Ratio
0.3816

Mammal Milk Dataset
K-Means  —  k = 2  [Appendix C-1]
python kmeans.py mammal_milk.csv 2  →  outputs/kmeans_mammal_k2.txt
K-Means with k=2 yields the best K-Means silhouette for this dataset (0.6251). The two clusters reflect a primary nutritional split: one group consists of species with high fat / low water content (marine mammals, some carnivores), and the other of species with lower fat and higher lactose (terrestrial herbivores). k=3 and k=4 fragment the clusters with reduced silhouette.
Metric
Value
Silhouette Score
0.6251
Cluster Sizes
Cluster 0: 17 | Cluster 1: 8
Radius/IC Ratio
0.6222

DBSCAN  —  ε = 5.0, minPts = 3  [Appendix C-2]
python dbscan.py mammal_milk.csv 5.0 3  →  outputs/dbscan_mammal_eps50_min3.txt
DBSCAN produced 4 clusters plus 3 outliers (12%) rather than the expected 2 groups. The additional fragmentation arises because certain species (e.g., those with extreme fat or protein values) form their own small dense pockets within the feature space at ε=5.0. The silhouette is 0.5516. Smaller ε values drove nearly all points to noise; there is no setting that cleanly produces 2 clusters without outliers.
Metric
Value
Silhouette Score
0.5516
Clusters
4 + 3 noise (12%)
Cluster Sizes
10 / 6 / 3 / 3 + 3 noise

Hierarchical Clustering  —  threshold = 100.0, single linkage  [Appendix C-3]
python hclustering.py mammal_milk.csv 100.0  →  outputs/hcluster_mammal_t100.txt
The highest silhouette for any Mammal Milk run (0.6888) comes from hierarchical clustering at threshold 100.0, which produces 2 clusters (sizes 23 and 2). The silhouette advantage comes from the extreme isolation of the 2-point cluster, which likely contains outlier species. This demonstrates single linkage's sensitivity to outlier bridges: a large threshold is needed to prevent the outlier from being chained into the main group.
Metric
Value
Silhouette Score
0.6888
Cluster Sizes
Cluster 0: 23 | Cluster 1: 2
Observation
Outlier species isolated in singleton-like cluster


Planets Dataset
K-Means  —  k = 2  [Appendix D-1]
python kmeans.py planets.csv 2  →  outputs/kmeans_planets_k2.txt
K-Means at k=2 achieves the highest silhouette score of any run across all datasets (0.7923), cleanly separating 17 inner/rocky planets from 2 outer giants. The large scale differences across planetary features (mass, orbital radius, etc.) strongly favor separation along a single dominant axis, making k=2 highly effective here.

Metric
Value
Silhouette Score
0.7923
Cluster Sizes
Cluster 0: 17 | Cluster 1: 2
Radius/IC Ratio
0.1903


DBSCAN  —  ε = 10.0, minPts = 3  [Appendix D-2]
python dbscan.py planets.csv 10.0 3  →  outputs/dbscan_planets_eps100_min3.txt
DBSCAN performs poorly on this dataset regardless of parameter settings. At ε=10.0, 8 of 19 points (42%) are classified as noise, and the two resulting clusters are fragmented. The fundamental issue is that unscaled planetary feature values span many orders of magnitude; the Euclidean distance metric conflates very different scales, and no single ε cleanly separates groups. Smaller ε drives all 19 points to noise; this is the least-bad setting found.

Metric
Value
Silhouette Score
0.2625 (worst among planets runs)
Outliers
8 (42.11%)
Cluster Sizes
Cluster 0: 4 | Cluster 1: 7

Hierarchical Clustering  —  threshold = 100.0, single linkage  [Appendix D-3]
python hclustering.py planets.csv 100.0  →  outputs/hcluster_planets_t100.txt
At threshold 100.0, our hierarchical implementation found 8 clusters (silhouette 0.5460). Sklearn's AgglomerativeClustering at the same threshold found only 2 clusters with silhouette 0.7923 — identical to K-Means. This is the largest discrepancy between our implementation and sklearn in this study. The difference is likely due to the sklearn implementation using a more efficient distance computation or different tie-breaking in cluster merging that avoids the chaining artifact. Our single-link implementation is susceptible to pulling points into intermediate clusters prematurely at this scale.

Metric
Value
Our Silhouette Score
0.5460 (8 clusters)
Sklearn Silhouette Score
0.7923 (2 clusters)
Discrepancy
Same threshold, same linkage — different # clusters


Accidents Dataset
K-Means  —  k = 3  [Appendix E-1]
python kmeans.py AccidentsSet03.csv 3  →  outputs/kmeans_accidents_k3.txt
K-Means at k=3 is the best K-Means result (0.4458), edging out k=2 (0.4100). The three clusters appear to be differentiated by combinations of severity-related features. This is the most challenging dataset, and silhouette scores are lower across all methods. k=4 further reduced silhouette to 0.4048.
Metric
Value
Silhouette Score
0.4458
Cluster Sizes
25 / 12 / 25
Radius/IC Ratio
1.6887 (high — clusters overlap)

DBSCAN  —  ε = 1.0, minPts = 3  [Appendix E-2]
python dbscan.py AccidentsSet03.csv 1.0 3  →  outputs/dbscan_accidents_eps10_min3.txt
DBSCAN at ε=1.0 finds 3 clusters plus 4 noise points (6.5%), with silhouette 0.4308. Larger ε values (5.0, 10.0) merged all points into a single cluster. The accidents data has sufficient density at ε=1.0 to form coherent groups, but the clusters do not represent dramatically distinct accident types.

Metric
Value
Silhouette Score
0.4308
Outliers
4 (6.45%)
Cluster Sizes
27 / 9 / 22 + 4 noise

Hierarchical Clustering  —  threshold = 2.0, single linkage  [Appendix E-3]
python hclustering.py AccidentsSet03.csv 2.0  →  outputs/hcluster_accidents_t2.txt
At threshold 2.0, our implementation produces 3 clusters with silhouette 0.3598 — the lowest result for this dataset. Sklearn's AgglomerativeClustering at the same threshold found 4 clusters (silhouette 0.3677). This discrepancy again reflects single-link chaining differences in implementation. The best hierarchical result for accidents (0.406 at threshold 1.0, 6 clusters) is still below the K-Means and DBSCAN results.
Metric
Value
Silhouette Score (t=2.0)
0.3598 (our) vs 0.3677 (sklearn)
Clusters
3 (our) vs 4 (sklearn)
Radius/IC Ratio
0.3582


Visualization
Iris Dataset
Figure 2 shows K-Means k=3 assignments plotted on the first two raw features (sepal length vs. sepal width). The purple cluster (setosa) is tightly packed in the upper-left and separates cleanly. The yellow (versicolor) and teal (virginica) clusters overlap in this 2D projection, reflecting the partial separability of these species across all four features. The centroids (X marks) are positioned accurately at the center of each cluster mass.

4Clusters Dataset
Figure 1 shows the 4clusters dataset with K-Means k=4 coloring. All four Gaussian blobs are clearly separated. Despite having four visually distinct groups, the silhouette score is higher at k=3 (0.6710 vs. 0.5699) because the two closest blobs (upper-left yellow and lower-right teal) have enough inter-cluster overlap that merging them raises average silhouette. The centroids (marked with X) sit near the geometric center of each blob.


Mammal Milk Dataset
Figure 3 shows K-Means k=2 on mammal milk. The yellow cluster (right side, high water %) corresponds to terrestrial herbivores with low fat and high water content. The purple cluster (left side, low water %) groups marine and carnivorous mammals with high fat. A handful of species in the middle range show why DBSCAN found additional sub-clusters; they sit between the two main density regions.

Planets Dataset
Figure 4 illustrates the extreme scale disparity in the planets dataset. Nearly all 17 inner/rocky planets are packed into the lower-left region of the plot, while the 2 outer gas giants sit far to the upper right at Feature 1 ≈ 340. This single dominant axis of separation explains K-Means' exceptional performance (silhouette 0.7923) and why DBSCAN fails: no single ε can simultaneously handle the dense inner-planet cluster and the isolated outer pair.

Accidents Dataset
Figure 5 shows K-Means k=3 on AccidentsSet03. The data is notably more compact than other datasets in this 2D projection — most points cluster near Feature 1 = 1–4, Feature 2 = 0–1. The three clusters partially overlap, which is consistent with the moderate silhouette score of 0.4458 and the high radius/intercluster ratio (1.69). The outlier at Feature 1 = 10 is absorbed into the purple cluster, illustrating how K-Means cannot flag noise the way DBSCAN does.


Discussion / Comparison to Scikit-Learn
Our sklearn comparison script (sklearn_compare.py) runs sklearn KMeans (n_init=10, random_state=0), DBSCAN, and AgglomerativeClustering with the same best parameters used in our implementations. The following observations highlight where results agree and where they diverge.
Iris
K-Means: Our implementation and sklearn produce identical silhouette scores (0.5526) and nearly identical Rand indices (0.8797). This agreement is notable because sklearn uses 10 restarts to avoid local optima — suggesting k=3 on Iris has a strong global optimum that our single random init reliably finds.
DBSCAN: Perfect agreement. Same ε, same core points, same cluster expansion — both produce 2 clusters + 3 noise points with silhouette 0.5016. The slight Rand index difference between our runs (0.7771) and sklearn’s report (0.7771) is identical.
H-Clustering: Both our implementation and sklearn found 2 clusters (not 3) at threshold 1.5 with single linkage, with identical silhouette 0.6864. This confirms correctness and also highlights the single-linkage chaining problem on Iris: the algorithm merges versicolor and virginica into one chain, leaving only setosa separate.
4clusters
K-Means: Our best result (k=3, sil=0.6710) differs from sklearn’s best-reported run (k=4, sil=0.5852). Sklearn reported k=4 as the “best parameter” but our grid search found k=3 yields a higher silhouette on this dataset. This reveals an important point: the “best k” from a silhouette perspective may not match the “true” number of clusters.
DBSCAN and H-Clustering: Both agree with sklearn at the same parameters. At ε=5.0, 6 outliers are labeled identically. The hierarchical 3-cluster partition at threshold 10.0 matches sklearn’s output exactly.
Mammal Milk
K-Means: Our implementation matches sklearn exactly (0.6251, cluster sizes 17/8). Sklearn’s n_init=10 does not produce a different result here, suggesting k=2 on this dataset has a stable global optimum.
DBSCAN: Both produce 4 clusters + 3 noise (silhouette 0.5516). The surprising finding of 4 rather than 2 clusters is consistent across implementations. This reflects genuine sub-structure in the data rather than an implementation artifact.
H-Clustering: Our implementation and sklearn agree perfectly (2 clusters, 23/2 split, silhouette 0.6888). The extreme skew in cluster sizes (23 vs 2) suggests a single-link chain collapsed almost all mammals into one cluster while isolating an outlier pair.
Planets
K-Means: Perfect agreement (silhouette 0.7923, cluster sizes 17/2). The strongest result in the study.
DBSCAN: Both implementations produce the same poor result (0.2625, 8 noise points). This is not an implementation failure — it is a genuine limitation of density-based clustering on unscaled astronomical data spanning orders of magnitude.
H-Clustering: This is the largest discrepancy in the study. Our implementation at threshold=100.0 produces 8 clusters (silhouette 0.5460), while sklearn’s AgglomerativeClustering at the same threshold and single linkage produces 2 clusters (silhouette 0.7923). We believe this discrepancy arises from how inter-cluster distances are computed when large absolute values are involved. Sklearn’s implementation uses optimized spatial indexing that may handle large-scale distance ties differently than our naive O(n³) merge loop.
Accidents
K-Means: Our implementation matches sklearn (silhouette 0.4458, cluster sizes 25/12/25). The relatively high silhouette for a noisy dataset suggests K-Means finds a defensible partition.
DBSCAN: Agreement is exact (0.4308, 4 noise points, 3 clusters). This is one of the better DBSCAN results relative to K-Means for any dataset in this study.
H-Clustering: Our implementation (threshold=2.0) produces 3 clusters with silhouette 0.3598, while sklearn produces 4 clusters with 0.3677. The difference is small in quality but involves an extra cluster, again pointing to implementation-level differences in single-link merge ordering for this dataset scale.
Key Observations
•       Single-linkage chaining: The most consistent problem across datasets. On Iris, it collapses 3 classes to 2 clusters. On 4clusters, it collapses 4 blobs to 3. On Mammal Milk at threshold 100.0, it isolates a 2-point outlier cluster. Complete or average linkage would likely produce more balanced clusters.
•       DBSCAN on unscaled data (Planets): DBSCAN is fundamentally unsuited to datasets with features that span orders of magnitude without normalization. The 42% noise rate is a direct consequence of this mismatch.
•       Silhouette vs. ground truth (Iris): The highest silhouette on Iris comes from hierarchical clustering with 2 clusters (0.6864), but the Rand index (0.7763) shows this does not match the 3-class biology. Silhouette is a useful guide but can reward merging that hides true structure.
•       k=3 beats k=4 on 4clusters: A higher silhouette for fewer clusters than the “true” number of blobs is a known pathology of silhouette-based model selection. It should always be paired with visual inspection and domain knowledge.
•       Planets H-clustering divergence: The largest unexplained discrepancy between our implementation and sklearn. This warrants further investigation into how single-link merging is performed at large absolute distances.


Analysis
Best result Analysis

Dataset
Algorithm
Best Parameters
Our Sil.
SK Sil.
Our Clusters
SK Clusters
Iris
K-Means
k = 3
0.5526
0.5526
3
3
Iris
DBSCAN
ε=0.7, minPts=4
0.5016
0.5016
2+noise(3)
2+noise(3)
Iris
H-Cluster (single)
threshold = 1.5
0.6864
0.6864
2 (!)
2 (!)
4clusters
K-Means
k = 3 (best sil)
0.6710
0.5852 (k=4)
3
4
4clusters
DBSCAN
ε=5.0, minPts=4
0.5432
0.5432
4+noise(6)
4+noise(6)
4clusters
H-Cluster (single)
threshold = 10.0
0.6710
0.6710
3
3
Mammal Milk
K-Means
k = 2
0.6251
0.6251
2
2
Mammal Milk
DBSCAN
ε=5.0, minPts=3
0.5516
0.5516
4+noise(3)
4+noise(3)
Mammal Milk
H-Cluster (single)
threshold = 100.0
0.6888
0.6888
2
2
Planets
K-Means
k = 2
0.7923
0.7923
2
2
Planets
DBSCAN
ε=10.0, minPts=3
0.2625
0.2625
2+noise(8)
2+noise(8)
Planets
H-Cluster (single)
threshold = 100.0
0.5460
0.7923
8
2
Accidents
K-Means
k = 3
0.4458
0.4458
3
3
Accidents
DBSCAN
ε=1.0, minPts=3
0.4308
0.4308
3+noise(4)
3+noise(4)
Accidents
H-Cluster (single)
threshold = 2.0
0.3598
0.3677
3
4


Algorithm Properties Comparison

Property
K-Means
H-Clustering (single link)
DBSCAN
Best For
Distance
Euclidean
Euclidean
Euclidean
All tested
Centroid / Merge
Mean of members
Greedy agglom.
N/A (density)
-
Params
k (# clusters)
distance threshold
ε and minPts
-
Handles Noise
No (absorbs)
No (absorbs)
Yes (label −1)
Noisy data: DBSCAN
Scalability
Fast O(nkd)
Slow O(n³)
O(n²) naive
Large: K-Means
Deterministic?
No (random init)
Yes
Yes
-
Best Dataset
Planets (0.7923)
Mammal Milk (0.6888)
Mammal Milk (0.5516)
-
Worst Dataset
Accidents (0.4458)
Accidents (0.3598)
Planets (0.2625)
-

Algorithm Properties Comparison
Metric
K-Means (k=3)
DBSCAN (ε=0.7, min=4)
H-Cluster (t=1.5)
Silhouette Score
0.5526
0.5016
0.6864
Rand Index
0.8797
0.7771
0.7763
# Clusters
3
2 + 3 noise pts
2
Radius/IC Ratio
0.8216
0.4854
0.4772

Best Overall Performance 
K-Means was the strongest and most consistent performer. It achieved the top silhouette score in 3 of 5 datasets (Iris, Mammal Milk, and Planets) and produced reliable, interpretable clusters with minimal tuning. Its primary weakness is random initialization, which can produce different results across runs, and its assumption of spherical, equal-size clusters.
Hierarchical Clustering (single linkage) achieved the highest silhouette on 2 datasets (Iris at 0.6864 and Mammal Milk at 0.6888), though both results involve subtle artifacts: the Iris result merges two true classes, and the Mammal Milk result isolates only 2 points in one cluster. Its dendrogram output provides unique exploratory value unavailable from the other methods.
DBSCAN was the weakest performer by silhouette in most datasets, but uniquely valuable for noise/outlier detection. Its results on Mammal Milk (correctly flagging anomalous species) and Accidents (flagging isolated records) are informative in ways K-Means and hierarchical clustering cannot provide.
Method-Dataset Fit
•       K-Means excelled on datasets with compact, roughly spherical clusters of similar scale: Iris, Planets (k=2), and Mammal Milk (k=2). It was least effective on Accidents, where clusters overlap significantly.
•       DBSCAN excelled when noise identification was valuable (Mammal Milk, Accidents) or clusters had non-convex shape. It was completely ineffective on unscaled Planets data.
•       H-Clustering excelled as an exploratory tool and for small datasets (Mammal Milk, Iris). Single linkage was consistently susceptible to chaining; average or complete linkage would likely perform better on datasets where the 4clusters and Iris problems arose.
Parameter Sensitivity	
All three algorithms are sensitive to their primary parameters, but in different ways. K-Means' k must be specified correctly; an incorrect k degrades silhouette substantially (e.g., Iris drops from 0.5526 at k=3 to 0.4171 at k=4). DBSCAN’s ε is highly scale-dependent: the range 0.5–0.7 works for Iris while 5.0–10.0 is needed for Mammal Milk and Planets requires even larger values. The hierarchical threshold exhibits the same scale dependency, spanning from 1.5 for Iris to 100.0 for Planets. Feature normalization before clustering would reduce this sensitivity.
Conclusion
No single clustering algorithm dominated across all five datasets. K-Means is the most reliable first-choice method for compact, well-structured data, but its inability to detect noise and its random initialization are practical limitations. DBSCAN is the right tool when outlier identification is important and when cluster shapes may be non-convex  but it requires feature normalization for datasets with heterogeneous scales. Hierarchical clustering provides unique structural insight through its dendrogram and is deterministic, but single-link chaining is a significant practical issue that would be resolved by switching to average or complete linkage.
The sklearn comparison confirmed our implementation correctness in most cases. The one meaningful discrepancy  Planets hierarchical clustering  suggests our O(n³) single-link implementation handles large absolute distance values differently than sklearn’s optimized implementation. Future work should include: K-Means++ initialization (or multi-restart), feature normalization for scale-heterogeneous datasets, and comparison of complete/average linkage to address single-link chaining.
Appendix — Clustering Output
Each run is identified by dataset, algorithm, and parameter settings. Output is generated by printReport() (clusterHelper.py) and printClusters() (hclustering.py). For runs with large point sets, only summary metrics and representative centroids are shown. Full verbatim output is in the outputs/ directory submitted with this report.
Appendix A — Iris Dataset
A-1: K-Means, k=3
outputs/kmeans_iris_k3.txt
•       150 data points | Clusters: [0, 1, 2]
•       Silhouette: 0.5526 | Rand Index: 0.8797 | Radius/IC Ratio: 0.8216
•       Cluster 0 (~50 pts): setosa — centroid ≈ [5.01, 3.43, 1.46, 0.25], tight and well-separated
•       Cluster 1 (~50 pts): versicolor — centroid ≈ [5.94, 2.77, 4.26, 1.33]
•       Cluster 2 (~50 pts): virginica — centroid ≈ [6.85, 3.07, 5.74, 2.07]
•       Converged in approximately 6–8 iterations
A-2: DBSCAN, ε=0.7, minPts=4
outputs/dbscan_iris_eps07_min4.txt
•       150 data points | Clusters: [-1, 0, 1]
•       Silhouette: 0.5016 | Rand Index: 0.7771 | Radius/IC Ratio: 0.4854
•       Outliers: 3 (2.00%)
•       Cluster 0: 50 pts (setosa) | Cluster 1: 97 pts (versicolor + virginica combined)
•       Note: ε=0.6 found 3 clusters but with more noise (5 pts); ε=0.5 had 13 noise pts and sil=0.381
A-3: H-Clustering, threshold=1.5, single linkage
outputs/hcluster_iris_t15.txt
•       150 data points | Clusters: [0, 1]
•       Silhouette: 0.6864 | Rand Index: 0.7763 | Radius/IC Ratio: 0.4772
•       Cluster 0: 100 pts (versicolor + virginica chained together) | Cluster 1: 50 pts (setosa)
•       Single-link chaining effect: versicolor/virginica boundary points bridge the two groups into one chain
•       Thresholds 2.0 and 2.5 merge all 150 pts into one cluster
Appendix B — 4clusters Dataset
B-1: K-Means, k=3 (best sil) and k=4 (true # clusters)
outputs/kmeans_4clusters_k3.txt | outputs/kmeans_4clusters_k4.txt
•       k=3: Silhouette 0.6710 | Radius/IC 0.3816 | Clusters [0,1,2]
•       k=4: Silhouette 0.5699 | Radius/IC 0.8700 | Clusters [0,1,2,3]
•       Two adjacent blobs merge at k=3; all four separate at k=4 but with lower silhouette
B-2: DBSCAN, ε=5.0, minPts=4
outputs/dbscan_4clusters_eps50_min4.txt
•       39 data points | Clusters: [-1, 0, 1, 2, 3]
•       Silhouette: 0.5432 | Outliers: 6 (15.38%) | Radius/IC: 0.4680
•       ε=8.0–10.0 yields 3 clusters, 0 noise, sil=0.6710 (same as k=3 result)
B-3: H-Clustering, threshold=10.0, single linkage
outputs/hcluster_4clusters_t10.txt
•       39 data points | Clusters: [0, 1, 2]
•       Silhouette: 0.6710 | Radius/IC: 0.3816
•       Threshold=5.0 produces 7 micro-clusters (sil=0.384); threshold=15.0+ merges to 1 cluster
Appendix C — Mammal Milk Dataset
C-1: K-Means, k=2
outputs/kmeans_mammal_k2.txt
•       25 data points | Clusters: [0, 1]
•       Silhouette: 0.6251 | Radius/IC: 0.6222
•       Cluster 0 (17 pts): terrestrial/herbivore mammals — lower fat, higher water/lactose
•       Cluster 1 (8 pts): marine/carnivore mammals — higher fat, lower water
C-2: DBSCAN, ε=5.0, minPts=3
outputs/dbscan_mammal_eps50_min3.txt
•       25 data points | Clusters: [-1, 0, 1, 2, 3]
•       Silhouette: 0.5516 | Outliers: 3 (12%) | Radius/IC: 0.3257
•       4 clusters + 3 noise points; sub-structure finer than k=2 expectation
•       ε=2.0 produces 11 noise points and sil=0.108; no intermediate setting yields clean 2 clusters
C-3: H-Clustering, threshold=100.0, single linkage
outputs/hcluster_mammal_t100.txt
•       25 data points | Clusters: [0, 1]
•       Silhouette: 0.6888 (best for Mammal Milk) | Radius/IC: 0.2707
•       Cluster 0: 23 pts | Cluster 1: 2 pts (outlier pair isolated by single linkage)
Appendix D — Planets Dataset
D-1: K-Means, k=2
outputs/kmeans_planets_k2.txt
•       19 data points | Clusters: [0, 1]
•       Silhouette: 0.7923 (highest in study) | Radius/IC: 0.1903
•       Cluster 0: 17 pts (inner/rocky) | Cluster 1: 2 pts (outer gas giants)
D-2: DBSCAN, ε=10.0, minPts=3
outputs/dbscan_planets_eps100_min3.txt
•       19 data points | Clusters: [-1, 0, 1]
•       Silhouette: 0.2625 | Outliers: 8 (42.11%) | Radius/IC: 0.1935
•       Cluster 0: 4 pts | Cluster 1: 7 pts | Noise: 8 pts
•       ε=5.0 drives 13 pts to noise (sil=−0.13); ε=1.0 drives all 19 to noise
D-3: H-Clustering, threshold=100.0, single linkage
outputs/hcluster_planets_t100.txt
•       19 data points | Clusters: [0,1,2,3,4,5,6,7] — 8 clusters
•       Silhouette: 0.5460 | Radius/IC: 0.2539
•       Sklearn at same params: 2 clusters, sil=0.7923 — significant divergence (see Section 4.4)
•       threshold=200.0 gives 5 clusters (sil=0.531); threshold=50.0 gives 12 clusters (sil=0.507)
Appendix E — Accidents Dataset (AccidentsSet03)
E-1: K-Means, k=3
outputs/kmeans_accidents_k3.txt
•       62 data points | Clusters: [0, 1, 2]
•       Silhouette: 0.4458 | Radius/IC: 1.6887 (high — clusters overlap in feature space)
•       Cluster sizes: 25 / 12 / 25
E-2: DBSCAN, ε=1.0, minPts=3
outputs/dbscan_accidents_eps10_min3.txt
•       62 data points | Clusters: [-1, 0, 1, 2]
•       Silhouette: 0.4308 | Outliers: 4 (6.45%) | Radius/IC: 0.8478
•       Cluster sizes: 27 / 9 / 22 + 4 noise
•       ε=5.0 and ε=10.0 produce a single all-encompassing cluster
E-3: H-Clustering, threshold=2.0, single linkage
outputs/hcluster_accidents_t2.txt
•       62 data points | Clusters: [0, 1, 2]
•       Silhouette: 0.3598 | Radius/IC: 0.3582
•       threshold=1.0: 6 clusters, sil=0.406 | threshold=5.0: 1 cluster | threshold=0.5: 29 near-singletons, sil=0.694 (trivial)
•       Sklearn at threshold=2.0: 4 clusters, sil=0.3677 — small discrepancy
Appendix F — Visualizations
Figures F-1 through F-5 are the K-Means scatter plots for each dataset at their best-performing k value.
 
F-1: Iris — K-Means k=3


F-2: 4clusters — K-Means k=4


 
F-3: Mammal Milk — K-Means k=2

F-4: Planets — K-Means k=2

 
F-5: Accidents — K-Means k=3

