# Phase 2: Capita Selecta Project

This folder includes scripts and experiment results of phase 2: Capita Project.
For full experiment pipeline, please refer to the [navigation guideline](https://github.com/anhtth16/xai_discretization-capita/blob/main/README.md) at the landing page.

Scope:
- Full experiment pipleline is performed on five datasets: adult, mustk, pageblock, phoneme, tranfusion
- Reperform training KNN using Hamming distance and bias-variance decomposition on some datasets in phase 1

Changes compared to phase 1:
- ChiMerge: not using manually defined function. Replace with Scorecard Bundle library
- KNN models: replace VDM distance by Hamming distance

Warning: 
- Bias-Variance decomposition takes very long time for calculation. For some datasets, we cannot run this evaluation for all models in one script, so it is advised to break down the whole scripts into smaller tasks.
- Srun scripts are Python scripts used to run in the HPC computing cluster.
