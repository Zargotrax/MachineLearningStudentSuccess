# Getting started
### necessary extensions
- Jupyter
- Python

## To work in VsCode
1. having python installed and `pip>=19.3` installed, run `pip install -r requirements.txt`
2. create a `venv` virtual environment (`ctrl+shift+p` -> `create environment`)
3. open `analyse_donnees.ipynb` and press `Run all`

## To work in JupyterLab
1. install [Anaconda](https://www.anaconda.com/download)
2. in the anaconda prompt, launch JupyterLab



# Main sections of the code
## root
At the root of the project, you will find the main analysis notebooks

### models_analysis.ipynb
An analysis of our models to select the best model for the SHAP analysis.

### analyse_shap_CatBoost.ipynb
A SHAP analysis of the best model (Catboost), according to our findings.

## models
In the models folder is the different models we tested as well as two agregation notebook to compare those models with the simplified dataset

## data
Original dataset as well as a simplified one and some related analysis and classes


