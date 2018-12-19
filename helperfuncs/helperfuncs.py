import pickle
import numpy as np
import matplotlib as mpl
mpl.use ('Agg')
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV

###############################################################################
# config datapath
###############################################################################
def get_datapath():
    return "./data/"
###############################################################################
# save and load experiments
###############################################################################
def save_experiment(results, name):
    file_name = get_datapath() + name + ".pkl"
    with open(file_name, "wb") as file:
        pickle.dump(results, file)

def load_experiment(name):
    file_name = get_datapath() + name + ".pkl"
    with open(file_name, "rb") as file:
        results = pickle.load(file)
    return results

###############################################################################
# Function to perform a grid search and return best model score
###############################################################################

def Gridsearch(model,param_grid,dataset,X,y,modelName):
    grid = GridSearchCV(model,param_grid, cv=10, refit=True, verbose=False,iid = False)
    grid.fit(X,y)
    bst_params = grid.best_params_
    save_experiment(bst_params, modelName + '_' + dataset + "_best_params")