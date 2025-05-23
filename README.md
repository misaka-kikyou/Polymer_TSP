# Polymer_TSP
Polymer Tensile Strength Prediction
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our station are listed below. To reproduce the machine learning results, please install packages with same version as below.

```plaintext
Package      | Version
-------------------------
Python       | 3.8.19
catboost     | 1.2.5
pandas       | 2.0.3
numpy        | 1.22.3
matplotlib   | 3.7.5
rdkit        | 2023.9.6
seaborn      | 0.13.2
scikit-learn | 1.3.0
scipy        | 1.10.1
tqdm         | 4.66.5
joblib       | 1.4.0
shap         | 0.44.1
lightgbm     | 4.5.0
optuna       | 4.0.0
xgboost      | 2.1.1
```

We suggest using Anaconda to build the python environment, as there are several default packages used in the scripts but not mentioned above. All test were executed under Windows 11 NT x64 10.0.22000 on Visual Studio Code 1.93.1.

Scatter plot comparing predicted versus experimental tensile strength values (MPa) for the XGBoost regression model. 
![image](https://github.com/user-attachments/assets/4bbab867-0d60-4c89-acd1-5fc369058ed5)

Feature correlation matrix (bubble plot) showing relationships between molecular descriptors. Red bubbles indicate positive correlations, blue bubbles indicate negative correlations, and bubble size represents correlation magnitude. 
![image](https://github.com/user-attachments/assets/0edc125c-16e4-4d5e-8521-8ca646085099)

