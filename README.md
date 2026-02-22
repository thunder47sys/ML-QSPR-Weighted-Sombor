# ML-QSPR-Weighted-Sombor

> A comparative machine learning framework for predicting physicochemical properties using 28 novel Weighted Sombor Indices.

## About This Project
This repository contains the data, computational results, and Python scripts for our research on Quantitative Structure-Property Relationship (QSPR) modeling. We introduce a "Physico-Topological" framework by modifying the geometric Sombor Index with four fundamental atomic weights: Atomic Mass, Atomic Radius, Electronegativity, and Ionization Energy. 

We benchmarked simple linear models against advanced non-linear machine learning algorithms (Random Forest, XGBoost, ANN, SVR) to predict six physicochemical properties for a diverse dataset of 200 pharmaceutical compounds.

## Repository Structure

### 1. `/data`
Contains the primary datasets used for training and testing.
* `Final 200_Compounds.xlsx`: The master dataset containing Canonical SMILES, compound names, and experimental properties.
* `Data_Alpha_Geometric_Weighted_Sombor_Indices.xlsx`: Contains the full matrix of 28 calculated Weighted Sombor Indices.

### 2. `/results`
Contains the mathematical outputs and statistical tables.
* `Clean_Unique_200_Pharmaceuticals.xlsx`: R2, RMSE, and MAE scores for all seven regression models.
* `Supplementary_Predicted_Values.csv`: A direct comparison of the actual vs. predicted values and the residual gaps.
* `Correlation_Analysis_Pearson_R.csv`: The Pearson correlation matrix between descriptors and properties.
* `Supplementary_All_Feature_Importances.csv`: The Gini impurity reduction scores from the tree-based models.

### 3. `/scripts`
Contains the core Python codes used to run the experiments.
* `indicescal.py`: Calculates the 28 Weighted Sombor Indices for the compounds.
* `models.py`: Evaluates the regression models and calculates performance metrics.
* `predicted.py`: Generates the predicted property values and residual gaps.
* `featureimp.py`: Extracts the feature importance tables for the models.
* `correlations.py`: Calculates the Pearson correlation scores.
* `vif.py`: Calculates the Variance Inflation Factor (VIF) scores.

## Usage Requirements
Please ensure all dependencies listed in the `requirements.txt` file are installed before running the scripts.
## Computational Environment
All mathematical modeling, data processing, and machine learning scripts were successfully executed and validated on the following setups:
* **Local Hardware:** HP ProBook 640 (Intel Core i7, 8th Generation).
* **Cloud Environment:** Google Colab (Python 3).

The scripts are highly optimized. They can run smoothly on standard local machines or be executed directly within Google Colab to ensure complete reproducibility without requiring heavy local resources.
