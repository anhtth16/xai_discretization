# Navigation Guidelines
There are two phases in this projects.
* Phase 1: baseline folder
* Phase 2: capita folder


Both phases follow the same pipeline:
STEP 1: Discretization data:
- Unsupervised discrezers: Equal Width Discretizer (EWD), Equal Frequency Discretizer (EFD), Fixed Frequency Discretizer (FFD)
STEP 2: Training models: 
Classical Naive Bayes, Decision Tree (ID3), and KNN models (KNN-VDM for phase 1, KNN-Hamming for phase 2)
STEP 3: Evaluation: Analysis performed in Phase 2 include output of both phase 1 and phase 2.
* Inconsistency rate: performed on the discretized datasets:
*  Bias-Variance decomposition: performed on the models after training
* Time for discretization, time for training models
* Wilcoxon signed rank tests for 3 metrics: accuracy, bias, variance: 
  - Two-sided test for phase 1, 
  - One-sided test for phase 2


