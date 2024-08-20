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
    
    name = f"{model}_spinemet_data"

    # Load and Preprocess
    df = pd.read_csv("Data/spinemet.tsv", sep = "\t")
    drop_list = ["patient_id"] # These are predictions...
    df = df.drop(drop_list,axis=1)
    time_scaler = MinMaxScaler()
    df["survival_in_days"]= time_scaler.fit_transform(df["survival_in_days"].to_numpy().reshape(-1, 1))
    t = np.float32(df["survival_in_days"].to_numpy())
    delta = df["censored"].to_numpy().astype(np.float32)
    # fix normal values, whenever available:
    normal_values = {"na":np.mean([136,145]),
                     "k":np.mean([3.5,5.1]),
                     "vvt":np.mean([3.8,5.8]),
                     "hg":np.mean([120,170]),
                     "htk":np.mean([35,50]),
                     "fvs":np.mean([4,10]),
                     "thr":np.mean([150,350]),
                     "creat":np.mean([53,88]),
                     "alp":150,
                     "ldh":160,
                     "albumin":np.mean([34,50]),
                     "serum_protein":np.mean([64,82])}
    df = df.fillna(normal_values)
    from preprocessing import Preprocessor
    cat_feats = ["sex","primer_tumor","histological_classification","surgery_season","invasiveness"]
    num_feats = ["age","paresis_scale","frankel_grade","preop_karnofsky","ecog",
                 "num_of_interspinal_metastases","num_of_operated_segments",
                 "avg_len_of_operated_segments","num_of_surgeries",
                 "num_of_extraspinal_bonemetastases","removability","hospital_days",
                 "asa","charlson_comorbidity_index","na","k","vvt","hg","htk",
                 "fvs","thr","creat","alp","ldh","albumin","serum_protein"]           
    pp = Preprocessor(cat_feat_strat="mode",num_feat_strat="knn",scaling_strategy="minmax",remaining="ignore")
    df = pp.fit_transform(df, cat_feats=cat_feats, num_feats=num_feats)
    X = df.drop(["survival_in_days","censored"],axis = 1).copy()
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