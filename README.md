# HPC-AI - Introduction to machine learning - Final project

The goal of this project is to predict whether a protein is part of a secretion system.

## Files description

### Python scripts

```project.py``` is the main script where we:
- import and pre-process the data
- train numerous ML pipelines
- chose one pipeline, valid and test it

```visualize.py``` is a secondary script used only to generate a visualization of our cross-validation strategy.

### Result folder

```parameters.csv``` contains all the different combination of parameters used to train the models

```scores.csv``` contains the corresponding scores (ROC AUC score, balanced accuracy and F2 score)

```test_pred.csv``` contains the labels predication for the test dataset

### PDF files
