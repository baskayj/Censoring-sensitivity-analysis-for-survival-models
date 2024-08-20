import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.random.set_random_seed(42)

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sksurv.metrics import concordance_index_ipcw, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from model import MDN
from utils import NLLLoss,CensoredNLLLoss,AlternativeNLLLoss
from utils import concordance_index_ipcw_scorer,integrated_brier_scorer,cumulative_dynamic_auc_scorer,root_mean_squared_error_scorer

import optuna

from optimizer import Optimizer
from sklearn.preprocessing import MinMaxScaler

import json

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.column import encode_categorical

name = "MDN_support2_data"

# Load and Preprocess
df = pd.read_csv("Data/support2.csv", index_col="id")
drop_list = ["sps","aps","surv2m","surv6m","prg2m","prg6m","dnr","dnrday"] # These are predictions...
df = df.drop(drop_list,axis=1)
time_scaler = MinMaxScaler()
df["d.time"]= time_scaler.fit_transform(df["d.time"].to_numpy().reshape(-1, 1))
t = np.float32(df["d.time"].to_numpy())
delta = df["death"].to_numpy().astype(np.float32)
# Fix Income to be numeric (there's ordering between the categories!!)
income = []
for value in df.income:
    if value == "under $11k":
        income.append(0.)
    elif value == "$11-$25k":
        income.append(1.)
    elif value == "$25-$50k":
        income.append(2.)
    elif value == ">$50k":
        income.append(3.)
    else:
        income.append(np.NaN)
df.income = income
# fix sfdm2 to be numeric (there's ordering between the categories!!)
sfdm2 = []
for value in df.sfdm2:
    if value == "no(M2 and SIP pres)":
        sfdm2.append(0.)
    elif value == "adl>=4 (>=5 if sur)":
        sfdm2.append(1.)
    elif value == "SIP>=30":
        sfdm2.append(2.)
    elif value == "Coma or Intub":
        sfdm2.append(3.)
    elif value == "<2 mo. follow-up":
        sfdm2.append(4.)
    else:
        sfdm2.append(np.NaN)
df.sfdm2 = sfdm2
# fix normal values, whenever available:
normal_values = {"alb":3.5,
                 "pafi":333.3,
                 "bili":1.01,
                 "crea":1.01,
                 "bun":6.51,
                 "wblc":9,
                 "urine":2502}
df = df.fillna(normal_values)
from preprocessing import Preprocessor
cat_feats = ["sex","hospdead","dzgroup","dzclass",
             "race","diabetes","dementia","ca"]
num_feats = ["age","slos","num.co","edu","income",
             "scoma","charges","avtisst","hday",
             "meanbp","wblc","hrt","resp","temp",
             "pafi","alb","bili","crea","sod",
             "bun","urine","sfdm2","adlsc"]
df = Preprocessor(cat_feat_strat="mode",num_feat_strat="knn",scaling_strategy="minmax").fit_transform(df, cat_feats=cat_feats, num_feats=num_feats)
X = df.copy()
x_size = len(X.columns)
X = np.float32(X.to_numpy())
y = np.stack([t,delta],axis = 1)

# Run Optimizer
print("Running Optimizer with LogRank binary scoring")
opt = Optimizer(X,y,name,num_epochs=200,batch_size=512,use_kfold=True,use_logrank=True,
                kernel_list = ["Exponential","Normal","Logistic","LogLogistic"])
best_val,best_params = opt(1000)
best_params['UnoC_LR']=best_val
print(best_params)
with open(f"Logs/{name}.json", "w") as write_file:
    json.dump(best_params, write_file)
    
print("Running Optimizer normaly")
opt = Optimizer(X,y,name,num_epochs=200,batch_size=512,use_kfold=True,use_logrank=False,
                kernel_list = ["Exponential","Normal","Logistic","LogLogistic"])
best_val,best_params = opt(1000)
best_params['UnoC']=best_val
print(best_params)
with open(f"Logs/{name}_no_logrank.json", "w") as write_file:
    json.dump(best_params, write_file)