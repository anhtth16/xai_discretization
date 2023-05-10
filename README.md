# Navigation Guidelines
There are two phases in this projects.
* Phase 1: [baseline folder](https://github.com/anhtth16/xai_discretization-capita/tree/main/baseline)
Result of phase 1 has been presented at BNAIC BeNeLearn 2022 Conference(https://bnaic2022.uantwerpen.be/)
* Phase 2: [capita folder](https://github.com/anhtth16/xai_discretization-capita/tree/main/capita): Extended experiment with new findings. The draft paper can be found in [this link](https://www.overleaf.com/read/xgshfwsfzsyj).


Both phases follow the same pipeline:

**STEP 1: Discretization data:**
Clean input data is discretized using different methods. The output will be used to train models (STEP 2)

* Unsupervised discrezers: Equal Width Discretizer (EWD), Equal Frequency Discretizer (EFD), Fixed Frequency Discretizer (FFD)
* Supervised discretizers:
- ChiMerge
- DecisionTree

**STEP 2: Training models:**
Three models are used for classification.
- Classical Naive Bayes, 
- Decision Tree (ID3), and 
- KNN models (KNN-VDM for phase 1, KNN-Hamming for phase 2)

Some datasets have imbalanced classes, thus, we performed SMOTE transformation before training the models (This issue happens mostly with datasets in capita phase)

**STEP 3: Evaluation:**

Analysis performed in Phase 2 include output of both phase 1 and phase 2.
* Inconsistency rate: performed on the discretized datasets:
*  Bias-Variance decomposition: performed on the models after training
* Time for discretization, time for training models
* Wilcoxon signed rank tests for 3 metrics: accuracy, bias, variance: 
  - Two-sided test for phase 1, 
  - One-sided test for phase 2


