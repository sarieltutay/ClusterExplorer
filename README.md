# ClusterExplorer
This repository contains the code for ClusterExplorer, a novel explainability tool for black-box clustering pipelines. Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters.

# Explaining the results of clustering pipelines
Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters. We achieve this by reducing the problem to generalized frequent-itemsets mining (gFIM), where items correspond to explanation predicates, and itemset frequency indicates coverage. To enhance efficiency, we leverage inherent problem properties and implement attribute selection to further reduce computational costs.

# Source Code
The source code is located in the [`ClusterExplorer/src`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src) directory. This directory contains the following key components:

1. **Explainer**:[`explainer.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/explainer.py) Generates rule-based explanations for each cluster using frequent-itemsets mining.

2. **Frequent Itemset Mining**:[`gFIM.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/gFIM.py) Contains methods for frequent itemset mining..

3. **Clustering Rule Evaluation**:[`ScoreMetrics.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/ScoreMetrics.py) [`AnalyzeItemsets.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/AnalyzeItemsets.py) Provides methods to evaluate and summarize the quality of clustering rules based on metrics such as separation error, coverage, and conciseness.

4. **Binning Methods**:[`binning_methods.py`](https://github.com/yourusername/ClusterExplorer/blob/main/src/binning_methods.py) Contains methods for binning numeric attributes, including equal width, equal frequency, decision tree-based, and multiclass optimal binning techniques.



