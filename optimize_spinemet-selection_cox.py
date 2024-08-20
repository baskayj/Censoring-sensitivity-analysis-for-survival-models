import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.random.set_random_seed(42)

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from sksurv.metrics import concordance_index_ipcw, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from model import MDN
from utils import NLLLoss,CensoredNLLLoss,AlternativeNLLLoss
from utils import concordance_index_censored_scorer,concordance_index_ipcw_scorer,integrated_brier_scorer,cumulative_dynamic_auc_scorer,root_mean_squared_error_scorer

import optuna

from optimizer import Optimizer,ML_Optimizer
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import json

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.column import encode_categorical

import sys

if __name__=='__main__':
    argv=sys.argv[1:] 
    kwargs={kw[0]:kw[1] for kw in [ar.split('=') for ar in argv if ar.find('=')>0]}
    args=[arg for arg in argv if arg.find('=')<0]
    model = kwargs.get("model","CoxPH")
    
    print(model)
    
    name = f"{model}_spinemet_selection_data"

    # Load and Preprocess
    df = pd.read_csv("Data/spinemet_selection.tsv", sep = "\t")
    drop_list = ["nmstpc","categories"] # These are predictions...
    df = df.drop(drop_list,axis=1)
    time_scaler = MinMaxScaler()
    df["survival"]= time_scaler.fit_transform(df["survival"].to_numpy().reshape(-1, 1))
    t = np.float32(df["survival"].to_numpy())
    delta = df["census"].to_numpy().astype(np.float32)
    from preprocessing import Preprocessor
    num_feats = ["primer_tumor","age","mobility","metastasis","protein"]        
    pp = Preprocessor(cat_feat_strat="mode",num_feat_strat="knn",scaling_strategy="minmax",remaining="drop")
    df = pp.fit_transform(df, cat_feats=[], num_feats=num_feats)
    X = df.copy()
    x_size = len(X.columns)
    X = np.float32(X.to_numpy())
    y = np.stack([t,delta],axis = 1)

    # Run Optimizer
    print("Running Optimizer with LogRank binary scoring")
    opt = ML_Optimizer(X,y,model=model,name=name,use_kfold=True,use_logrank=True)
    best_val,best_params = opt(1000)
    best_params['UnoC_LR']=best_val
    print(best_params)
    with open(f"Logs/{name}.json", "w") as write_file:
        json.dump(best_params, write_file)
        
    print("Running Optimizer normaly")
    opt = ML_Optimizer(X,y,model=model,name=name,use_kfold=True,use_logrank=False)
    best_val,best_params = opt(1000)
    best_params['UnoC']=best_val
    print(best_params)
    with open(f"Logs/{name}_no_logrank.json", "w") as write_file:
        json.dump(best_params, write_file)