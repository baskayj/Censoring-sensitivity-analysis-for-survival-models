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
    
    name = f"{model}_metabric_data"

    # Load and Preprocess
    df = pd.read_csv("Data/metabric.tsv",sep="\t",header=4).drop(["PATIENT_ID","VITAL_STATUS","RFS_MONTHS"],axis=1)
    cell = []
    for c in df.CELLULARITY:
        if c == "Low":
            cell.append(0.0)
        elif c == "Moderate":
            cell.append(0.5)
        elif c == "High":
            cell.append(1.0)
        else:
            cell.append(np.NaN)
    df.CELLULARITY = cell
    df = df.dropna()
    df = encode_categorical(df)
    cont_features = ["LYMPH_NODES_EXAMINED_POSITIVE","NPI","CELLULARITY","AGE_AT_DIAGNOSIS"]
    scaler = MinMaxScaler()
    df[cont_features] = scaler.fit_transform(df[cont_features])
    time_scaler = MinMaxScaler()
    df.OS_MONTHS = time_scaler.fit_transform(df.OS_MONTHS.to_numpy().reshape(-1, 1))
    X = df.drop(["OS_MONTHS","OS_STATUS=1:DECEASED"],axis = 1)
    x_size = len(X.columns)
    X = np.float32(X.to_numpy())
    t = np.float32(df.OS_MONTHS.to_numpy())
    delta = df["OS_STATUS=1:DECEASED"].to_numpy().astype(bool)
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