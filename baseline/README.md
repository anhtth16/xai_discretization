# Phase 1: Baseline

This folder contain source code of the baseline study.

Scope of experiments:
- Datasets: iris, satimage, australia, pima, pendigits
- Discretization methods: Supervised discretizer (ChiMerge, DecisionTree) & Unsupervised discretizer (EWD, EFD, FFD)
- Classification models: Classical Naive Bayes, Decision Tree (ID3), KNN models using Value Difference Metric for the distance.
- Total: 270 models

Evaluation:
  - Model performace: Accuracy, Bias, Variace
  - Inconsistency of dataset after discretization
  - Computing time for each process: discretization and training models

The paper was presented at [BNAIC - BeNeLearn Conference 2022](https://bnaic2022.uantwerpen.be/wp-content/uploads/BNAICBeNeLearn_2022_submission_8652.pdf)
