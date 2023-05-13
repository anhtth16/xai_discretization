# Phase 1: Baseline

This folder contain source code of the baseline study.

Scope of experiments:

- Datasets: iris, satimage, australia, pima, pendigits. Clean input data before discretization is in folder [clean-input-datasets](https://github.com/anhtth16/xai_discretization/tree/main/baseline/clean-input-datasets)

- Discretization methods: 
	- Supervised discretizer (ChiMerge, DecisionTree): See folder [Supervised Discretization](https://github.com/anhtth16/xai_discretization/tree/main/baseline/Supervised%20Discretization)
	- Unsupervised discretizer (EWD, EFD, FFD): See folder [Unsupervised Discretization](https://github.com/anhtth16/xai_discretization/tree/main/baseline/Unsupervised%20Discretization)
	
- Classification models: Each dataset after discretized is used to train classification models. We use three models:
	- Classical Naive Bayes
	- Decision Tree (ID3)
	- KNN models using Value Difference Metric for the distance.

- Total: 270 models

Evaluation:

- Inconsistency of dataset after discretization (See folder [Inconsistency](https://github.com/anhtth16/xai_discretization/tree/main/baseline/Inconsistency))
- Model performace: Accuracy, Bias, Variace
- Computing time for each process: discretization and training models
- Wilconxon two-sided tests for three variables
- High resolution figures are in folder [BNAIC-figures](https://github.com/anhtth16/xai_discretization/tree/main/baseline/bnaic-figures)

The paper was presented at [BNAIC - BeNeLearn Conference 2022](https://bnaic2022.uantwerpen.be/wp-content/uploads/BNAICBeNeLearn_2022_submission_8652.pdf)
