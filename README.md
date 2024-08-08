# ClusterExplorer
This repository contains the code for ClusterExplorer, a novel explainability tool for black-box clustering pipelines. Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters.

# Explaining the results of clustering pipelines
Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters. We achieve this by reducing the problem to generalized frequent-itemsets mining (gFIM), where items correspond to explanation predicates, and itemset frequency indicates coverage. To enhance efficiency, we leverage inherent problem properties and implement attribute selection to further reduce computational costs.

# Source Code
The source code is located here (ClusterExplorer/src):
 Under this directory, there is the code for:
 1. Explainar which generating rule-based explanations for each cluster.
 2. Contains methods for frequent itemset mining.
 3. Methods to evaluate and summarize the quality of clustering rules based on metrics such as separation error, coverage, and conciseness.
 4. Methods for binning numeric attributes, including equal width, equal frequency, decision tree-based, and multiclass optimal binning techniques.

## Source Code
The source code is located in the [`ClusterExplorer/src`](https://github.com/sarieltutay/ClusterExplorer/edit/main/README.md) directory. This directory contains the following key components:

1. **Explainer**:
   - File: [`explainer.py`](https://github.com/yourusername/ClusterExplorer/blob/main/src/explainer.py)
   - Functionality: Generates rule-based explanations for each cluster using frequent-itemsets mining.

2. **Frequent Itemset Mining**:
   - File: [`gFIM.py`](https://github.com/yourusername/ClusterExplorer/blob/main/src/gFIM.py)
   - Functionality: Contains methods for frequent itemset mining..

3. **Clustering Rule Evaluation**:
   - File: [`ScoreMetrics.py`](https://github.com/yourusername/ClusterExplorer/blob/main/src/ScoreMetrics.py)
   - Functionality: Provides methods to evaluate and summarize the quality of clustering rules based on metrics such as separation error, coverage, and conciseness.

4. **Binning Methods**:
   - File: [`binning_methods.py`](https://github.com/yourusername/ClusterExplorer/blob/main/src/binning_methods.py)
   - Functionality: Contains methods for binning numeric attributes, including equal width, equal frequency, decision tree-based, and multiclass optimal binning techniques.



