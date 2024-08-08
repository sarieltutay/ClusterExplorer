# ClusterExplorer
This repository contains the code for ClusterExplorer, a novel explainability tool for black-box clustering pipelines. Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters.

## Explaining the results of clustering pipelines
Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters. We achieve this by reducing the problem to generalized frequent-itemsets mining (gFIM), where items correspond to explanation predicates, and itemset frequency indicates coverage. To enhance efficiency, we leverage inherent problem properties and implement attribute selection to further reduce computational costs.

## Source Code
The source code is located in the [`ClusterExplorer/src`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src) directory. This directory contains the following key components:

1. **Explainer**:[`explainer.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/explainer.py) Generates rule-based explanations for each cluster using frequent-itemsets mining.

2. **Frequent Itemset Mining**:[`gFIM.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/gFIM.py) Contains methods for frequent itemset mining..

3. **Clustering Rule Evaluation**:[`ScoreMetrics.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/ScoreMetrics.py) [`AnalyzeItemsets.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/AnalyzeItemsets.py) Provides methods to evaluate and summarize the quality of clustering rules based on metrics such as separation error, coverage, and conciseness.

4. **Binning Methods**:[`binning_methods.py`](https://github.com/yourusername/ClusterExplorer/blob/main/src/binning_methods.py) Contains methods for binning numeric attributes, including equal width, equal frequency, decision tree-based, and multiclass optimal binning techniques.

## Experiment Datasets
Cluster-Explorer was evaluated using a diverse set of 98 clustering results obtained from various clustering pipelines and algorithms. The datasets used in these experiments were sourced from the UCI Machine Learning Repository and cover a wide range of data shapes and sizes.

### Datasets Overview
The datasets used in the experiments include:

1. **Urban Land Cover**: 168 rows, 148 attributes
2. **DARWIN**: 174 rows, 451 attributes
3. **Wine**: 178 rows, 13 attributes
4. **Flags**: 194 rows, 30 attributes
5. **Parkinson Speech**: 1,040 rows, 26 attributes
6. **Communities and Crime**: 1,994 rows, 128 attributes
7. **Turkiye Student Evaluation**: 5,820 rows, 33 attributes
8. **In-vehicle Coupon Recommendation**: 12,684 rows, 23 attributes
9. **Human Activity Recognition**: 10,299 rows, 561 attributes
10. **Quality Assessment of Digital Colposcopies**: 30,000 rows, 23 attributes
11. **RT-IoT2022**: 123,117 rows, 85 attributes
12. **Gender by Name**: 147,270 rows, 4 attributes
13. **Multivariate Gait Data**: 181,800 rows, 7 attributes
14. **Wave Energy Converters**: 288,000 rows, 49 attributes
15. **3D Road Network**: 434,874 rows, 4 attributes
16. **Year Prediction MSD**: 515,345 rows, 90 attributes
17. **Online Retail**: 1,067,371 rows, 8 attributes
18. **MetroPT-3 Dataset**: 1,516,948 rows, 15 attributes
19. **Taxi Trajectory**: 1,710,670 rows, 9 attributes

### Clustering Pipelines
The clustering results were generated using 16 different clustering pipelines, each combining various preprocessing steps and clustering algorithms. The preprocessing steps included standard scaling for numeric columns, one-hot encoding for categorical data, and dimensionality reduction using PCA. The clustering algorithms used were K-Means, DBSCAN, Birch, Spectral Clustering, and Affinity Propagation.





