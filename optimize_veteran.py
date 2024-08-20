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

name = "MDN_veteran_data"

# Load and Preprocess
df,y = load_veterans_lung_cancer()
df["Status"] = y["Status"]
df["Survival_in_days"] = y["Survival_in_days"]
df = encode_categorical(df)
df = pd.DataFrame(MinMaxScaler().fit_transform(df),columns=df.columns)
X = df.drop(["Survival_in_days","Status"],axis = 1)
x_size = len(X.columns)
X = np.float32(X.to_numpy())
t = np.float32(df.Survival_in_days.to_numpy())
delta = df.Status.to_numpy().astype(np.float32)
y = np.stack([t,delta],axis = 1)

# Run Optimizer
print("Running Optimizer with LogRank binary scoring")
opt = Optimizer(X,y,name,num_epochs=400,batch_size=16,use_kfold=True,use_logrank=True)
best_val,best_params = opt(1000)
best_params['UnoC_LR']=best_val
print(best_params)
with open(f"Logs/{name}.json", "w") as write_file:
    json.dump(best_params, write_file)
    
print("Running Optimizer normaly")
opt = Optimizer(X,y,name,num_epochs=400,batch_size=16,use_kfold=True,use_logrank=False)
best_val,best_params = opt(1000)
best_params['UnoC']=best_val
print(best_params)
with open(f"Logs/{name}_no_logrank.json", "w") as write_file:
    json.dump(best_params, write_file)