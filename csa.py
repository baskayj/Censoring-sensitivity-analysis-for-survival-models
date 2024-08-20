import tensorflow as tf
tf.compat.v1.enable_eager_execution()

tf.compat.v1.random.set_random_seed(42)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import MDN
from utils import NLLLoss, CensoredNLLLoss, AlternativeNLLLoss

import numpy as np
import matplotlib.pyplot as plt

import os
import json

# Models
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

# Metrics
from utils import log_rank_test, log_rank_test_scorer
from utils import concordance_index_censored_scorer,concordance_index_ipcw_scorer,integrated_brier_scorer,cumulative_dynamic_auc_scorer
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw, integrated_brier_score, cumulative_dynamic_auc

# Utility
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from utils import reduce_T_max,reduce_uncensored


class MLCensoringSensitivityAnalysis:
    def __init__(self,
                 Model,
                 model_name,
                 X,
                 y,
                 Optimized_LogRank,
                 **kwargs):
        #super(MLCensoringSensitivityAnalysis, self).__init__(name="MLCSA")
        self.Model = Model
        self.model_name = model_name
        self.X = X
        self.y = y
        self.Optimized_LogRank = Optimized_LogRank
        self.kwargs = kwargs

    def ml_kfold_scorer(self,X,y,red_rate):
        kf = KFold(n_splits=5,random_state=42,shuffle=True)
        UnoCs = []
        HarrelCs = []
        iBriers = []
        AUROCs = []
        RMSEs = []
        LOGRs = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train = reduce_uncensored(y_train, red_rate)
            # Filter the test set for t > max(t_train), as these can cause problems for the scores (...)
            t_max = min([max(y_train[:,0]),max(y_test[:,0])])
            t_min = max([min(y_train[:,0]),min(y_test[:,0])])
            mask = (t_min < y_test[:,0]) & (y_test[:,0] < t_max)
            X_test = X_test[mask]
            y_test = y_test[mask]
            t_max = min([t_max,max(y_train[:,0]),max(y_test[:,0])])
            t_min = max([t_min,min(y_train[:,0]),min(y_test[:,0])])
            t, delta = y_train[:,0],y_train[:,1].astype(bool)
            survival_train = Surv().from_arrays(delta,t)
            t, delta = y_test[:,0],y_test[:,1].astype(bool)
            survival_test = Surv().from_arrays(delta,t)

            model = self.Model(**self.kwargs)
            model.fit(X_train,survival_train)
            unoc = concordance_index_ipcw(survival_train, survival_test, model.predict(X_test))[0]
            harrelc = concordance_index_censored(delta.astype(bool), t, model.predict(X_test))[0]
            survivals = model.predict_survival_function(X_test)
            mask = (t_min < survivals[0].x) & (survivals[0].x < t_max)
            timeline = survivals[0].x[mask]
            survs = []
            for survival in survivals:
                survs.append(survival.y[mask])
            ibrier = integrated_brier_score(survival_train,survival_test,survs,timeline)
            logr = log_rank_test(survs,timeline,y_test)
            # For Cox and Boosting we don't need the hazard functions (look at scikit-survival's demo to understand...)
            if self.Model in [CoxPHSurvivalAnalysis,GradientBoostingSurvivalAnalysis]:
                auroc = np.nanmean(cumulative_dynamic_auc(survival_train,survival_test,model.predict(X_test),timeline)[0])
            else:
                hazards = model.predict_cumulative_hazard_function(X_test)
                mask = (t_min < survivals[0].x) & (survivals[0].x < t_max)
                timeline = hazards[0].x[mask]
                hazs = []
                for hazard in hazards:
                    hazs.append(hazard.y[mask])
                auroc = np.nanmean(cumulative_dynamic_auc(survival_train,survival_test,hazs,timeline)[0])
            UnoCs.append(unoc)
            HarrelCs.append(harrelc)
            iBriers.append(ibrier)
            AUROCs.append(auroc)
            LOGRs.append(logr)
        HarrelCs = np.array(HarrelCs)
        UnoCs = np.array(UnoCs)
        iBriers = np.array(iBriers)
        AUROCs = np.array(AUROCs)
        LOGRs = np.array(LOGRs)
        #print(HarrelCs)
        #print(UnoCs)
        #print(iBriers)
        #print(AUROCs)
        #print(LOGRs)
        return {"HarrelC":{"mean":np.mean(HarrelCs),"std":np.std(HarrelCs)},
                "UnoC":{"mean":np.mean(UnoCs),"std":np.std(UnoCs)},
                "iBrier":{"mean":np.mean(iBriers),"std":np.std(iBriers)},
                "AUROC":{"mean":np.mean(AUROCs),"std":np.std(AUROCs)},
                "LogRank":{"mean":np.mean(LOGRs),"std":np.std(LOGRs)}}

    def ml_holdout_scorer(self,X_train,X_test,y_train,y_test,red_rate):
        UnoC = np.NaN
        HarrelC = np.NaN
        iBrier = np.NaN
        AUROC = np.NaN
        RMSE = np.NaN
        LOGR = np.NaN
        Timeline = []
        Overall_Survival_Function = []

        # Evaluate model performance on Hold-out set
        y_train = reduce_uncensored(y_train, red_rate)

        # Filter the test set for t > max(t_train), as these can cause problems for the scores (...)
        t_max = min([max(y_train[:,0]),max(y_test[:,0])])
        t_min = max([min(y_train[:,0]),min(y_test[:,0])])
        mask = (t_min < y_test[:,0]) & (y_test[:,0] < t_max)
        X_test = X_test[mask]
        y_test = y_test[mask]
        t_max = min([t_max,max(y_train[:,0]),max(y_test[:,0])])
        t_min = max([t_min,min(y_train[:,0]),min(y_test[:,0])])
        t, delta = y_train[:,0],y_train[:,1].astype(bool)
        survival_train = Surv().from_arrays(delta,t)
        t, delta = y_test[:,0],y_test[:,1].astype(bool)
        survival_test = Surv().from_arrays(delta,t)

        model = self.Model(**self.kwargs)
        model.fit(X_train,survival_train)
        HarrelC = concordance_index_censored(delta.astype(bool), t, model.predict(X_test))[0]
        UnoC = concordance_index_ipcw(survival_train, survival_test, model.predict(X_test))[0]
        survivals = model.predict_survival_function(X_test)
        mask = (t_min < survivals[0].x) & (survivals[0].x < t_max)
        timeline = survivals[0].x[mask]
        survs = []
        for survival in survivals:
            survs.append(survival.y[mask])
        iBrier = integrated_brier_score(survival_train,survival_test,survs,timeline)
        LOGR = log_rank_test(survs,timeline,y_test)

        Timeline = timeline.tolist()
        prob_survival = np.sum(np.array(survs),axis = 0)/len(X_test)
        Overall_Survival_Function = prob_survival.tolist()

        # Plots
        km_timeline, km_prob_survival = kaplan_meier_estimator(y_test[:,1].astype(bool),y_test[:,0])
        plt.plot(timeline,prob_survival, label = self.model_name)
        plt.plot(km_timeline,km_prob_survival, label = "KM")
        plt.legend()
        plt.show()

        if self.Model in [CoxPHSurvivalAnalysis,GradientBoostingSurvivalAnalysis]:
            AUROC = np.nanmean(cumulative_dynamic_auc(survival_train,survival_test,model.predict(X_test),timeline)[0])
        else:
            hazards = model.predict_cumulative_hazard_function(X_test)
            mask = (t_min < survivals[0].x) & (survivals[0].x < t_max)
            timeline = hazards[0].x[mask]
            hazs = []
            for hazard in hazards:
                hazs.append(hazard.y[mask])
            AUROC = np.nanmean(cumulative_dynamic_auc(survival_train,survival_test,hazs,timeline)[0])


        #print(HarrelC)
        #print(UnoC)
        #print(iBrier)
        #print(AUROC)
        #print(LOGR)
        return {"HarrelC" : HarrelC,
                "UnoC" : UnoC,
                "iBrier" : iBrier,
                "AUROC" : AUROC,
                "LogRank" : LOGR,
                "Timeline" : Timeline,
                "Overall_Survival_Function" : Overall_Survival_Function}

    def analyze(self,NAME,sensitivity_grid):
        for T_max, red_rate in sensitivity_grid:
            print(f"T_max:{T_max},red_rate:{red_rate}")

            # Open the results table
            if os.path.exists(f"Logs/{NAME}_results.json"):
                with open(f"Logs/{NAME}_results.json", "r") as read_file:
                    results_table = json.load(read_file)
            # If it doesn't exist create one
            else:
                results_table = {"Model":[],
                                 "Optimized_LogRank":[],
                                 "T_max":[],
                                 "Uncensored_Reduction":[],
                                    "5Fold":{"HarrelC":{"mean":[],
                                                        "std":[]},
                                             "UnoC":{"mean":[],
                                                     "std":[]},
                                             "iBrier":{"mean":[],
                                                       "std":[]},
                                             "AUROC":{"mean":[],
                                                      "std":[]},
                                             "LogRank":{"mean":[],
                                                        "std":[]}},
                                    "Hold-out":{"HarrelC":[],
                                                "UnoC":[],
                                                "iBrier":[],
                                                "AUROC":[],
                                                "LogRank":[]},
                                 "Timeline":[],
                                 "Overall_Survival_Function":[]}
                with open(f"Logs/{NAME}_results.json", "w") as write_file:
                        json.dump(results_table, write_file)

            # Add properties of experiment
            results_table["Model"].append(self.model_name)
            results_table["Optimized_LogRank"].append(self.Optimized_LogRank)
            results_table["T_max"].append(T_max)
            results_table["Uncensored_Reduction"].append(red_rate)

            # Apply T_max reduction
            y_red = np.copy(self.y)
            y_red = reduce_T_max(y_red,T_max)

            # Split train-holdout sets
            X_train,X_test,y_train,y_test = train_test_split(self.X,y_red,test_size=0.3,random_state=42)

            # 5Fold validation
            scores = self.ml_kfold_scorer(X_train,
                                          y_train,
                                          red_rate)
            results_table["5Fold"]["HarrelC"]["mean"].append(scores["HarrelC"]["mean"])
            results_table["5Fold"]["HarrelC"]["std"].append(scores["HarrelC"]["std"])
            results_table["5Fold"]["UnoC"]["mean"].append(scores["UnoC"]["mean"])
            results_table["5Fold"]["UnoC"]["std"].append(scores["UnoC"]["std"])
            results_table["5Fold"]["iBrier"]["mean"].append(scores["iBrier"]["mean"])
            results_table["5Fold"]["iBrier"]["std"].append(scores["iBrier"]["std"])
            results_table["5Fold"]["AUROC"]["mean"].append(scores["AUROC"]["mean"])
            results_table["5Fold"]["AUROC"]["std"].append(scores["AUROC"]["std"])
            results_table["5Fold"]["LogRank"]["mean"].append(scores["LogRank"]["mean"])
            results_table["5Fold"]["LogRank"]["std"].append(scores["LogRank"]["std"])

            # Evaluate model performance on Hold-out set
            results = self.ml_holdout_scorer(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             red_rate)
            results_table["Hold-out"]["HarrelC"].append(results["HarrelC"])
            results_table["Hold-out"]["UnoC"].append(results["UnoC"])
            results_table["Hold-out"]["iBrier"].append(results["iBrier"])
            results_table["Hold-out"]["AUROC"].append(results["AUROC"])
            results_table["Hold-out"]["LogRank"].append(results["LogRank"])
            results_table["Timeline"].append(results["Timeline"])
            results_table["Overall_Survival_Function"].append(results["Overall_Survival_Function"])

            # Save results for each set-up
            with open(f"Logs/{NAME}_results.json", "w") as write_file:
                json.dump(results_table, write_file)


class MDNCensoringSensitivityAnalysis:
    def __init__(self,
                 Model,
                 model_name,
                 dataset_name,
                 X,
                 y,
                 Optimized_LogRank,
                 **kwargs):
        self.Model = Model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.X = X
        self.y = y
        self.Optimized_LogRank = Optimized_LogRank
        self.kwargs = kwargs

    def mdn_kfold_scorer(self,
                         n_hidden,
                         mixture_components,
                         use_sparse_layer,
                         use_sparse_loss,
                         lmbd,
                         use_batchnorm,
                         use_dropout,
                         dropout,
                         mlp_size_1,
                         mlp_size_2,
                         mlp_size_3,
                         kernel,
                         loss_name,
                         num_epochs,
                         batch_size,
                         input_shape,
                         learning_rate,
                         X,
                         y,
                         red_rate,
                         timeline_resolution):
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        UnoCs = []
        HarrelCs = []
        iBriers = []
        AUROCs = []
        RMSEs = []
        LOGRs = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train = reduce_uncensored(y_train, red_rate)
            # Filter the test set for t > max(t_train), as these can cause problems for the scores (...)
            t_max = min([max(y_train[:,0]),max(y_test[:,0])])
            t_min = max([min(y_train[:,0]),min(y_test[:,0])])
            mask = (t_min < y_test[:,0]) & (y_test[:,0] < t_max)
            X_test = X_test[mask]
            y_test = y_test[mask]
            t_max = min([t_max,max(y_test[:,0])])
            t_min = max([t_min,min(y_test[:,0])])
            timeline = np.linspace(t_min,t_max,timeline_resolution,endpoint=False).astype(np.float32)
            model = self.Model(input_shape,
                               n_hidden = n_hidden,
                               mixture_components = mixture_components,
                               use_sparse_layer = use_sparse_layer,
                               use_sparse_loss = use_sparse_loss,
                               lmbd = lmbd,
                               use_batchnorm = use_batchnorm,
                               use_dropout = use_dropout,
                               dropout = dropout,
                               mlp_size=(mlp_size_1,mlp_size_2,mlp_size_3),
                               kernel=kernel)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            loss = loss_name(model)
            model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
            callbacks = [EarlyStopping(patience=50,verbose=0,restore_best_weights=True),
                         ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=0),
                         ModelCheckpoint(f"Logs/model-mdn-{self.dataset_name}-kfold.h5", verbose=0, save_best_only=True, save_weights_only=True)]
            history = model.fit(X_train,y_train,epochs=num_epochs,callbacks=callbacks,validation_data=(X_test,y_test),batch_size=batch_size,verbose=0)
            model.load_weights(f"Logs/model-mdn-{self.dataset_name}-kfold.h5")
            os.remove(f"Logs/model-mdn-{self.dataset_name}-kfold.h5")
            unoc = concordance_index_ipcw_scorer(model = model, timeline = timeline, threshold = 0, y_train = y_train, y_test = y_test, X_test = X_test)
            harrelc = concordance_index_censored_scorer(model = model, timeline = timeline, threshold = 0, y_test = y_test, X_test = X_test)
            ibrier = integrated_brier_scorer(model = model, timeline = timeline, threshold = 0, y_train = y_train, y_test = y_test, X_test = X_test)
            auroc = cumulative_dynamic_auc_scorer(model = model, timeline = timeline, threshold = 0, y_train = y_train, y_test = y_test, X_test = X_test)
            logr = log_rank_test_scorer(model = model, timeline = timeline, threshold=  0, y_test = y_test, X_test = X_test)
            UnoCs.append(unoc)
            HarrelCs.append(harrelc)
            iBriers.append(ibrier)
            AUROCs.append(auroc)
            LOGRs.append(logr)
        HarrelCs = np.array(HarrelCs)
        UnoCs = np.array(UnoCs)
        iBriers = np.array(iBriers)
        AUROCs = np.array(AUROCs)
        LOGRs = np.array(LOGRs)
        #print(HarrelCs)
        #print(UnoCs)
        #print(iBriers)
        #print(AUROCs)
        #print(LOGRs)
        return {"HarrelC":{"mean":np.mean(HarrelCs),"std":np.std(HarrelCs)},
                "UnoC":{"mean":np.mean(UnoCs),"std":np.std(UnoCs)},
                "iBrier":{"mean":np.mean(iBriers),"std":np.std(iBriers)},
                "AUROC":{"mean":np.mean(AUROCs),"std":np.std(AUROCs)},
                "LogRank":{"mean":np.mean(LOGRs),"std":np.std(LOGRs)}}

    def mdn_holdout_scorer(self,
                          n_hidden,
                          mixture_components,
                          use_sparse_layer,
                          use_sparse_loss,
                          lmbd,
                          use_batchnorm,
                          use_dropout,
                          dropout,
                          mlp_size_1,
                          mlp_size_2,
                          mlp_size_3,
                          kernel,
                          loss_name,
                          num_epochs,
                          batch_size,
                          input_shape,
                          learning_rate,
                          X_train,
                          X_test,
                          y_train,
                          y_test,
                          red_rate,
                          timeline_resolution):
        UnoC = np.NaN
        HarrelC = np.NaN
        iBrier = np.NaN
        AUROC = np.NaN
        RMSE = np.NaN
        LOGR = np.NaN
        Timeline = []
        Overall_Survival_Function = []

        # Evaluate model performance on Hold-out set
        y_train = reduce_uncensored(y_train, red_rate)
        # Filter the test set for t > max(t_train), as these can cause problems for the scores (...)
        t_max = min([max(y_train[:, 0]), max(y_test[:, 0])])
        t_min = max([min(y_train[:, 0]), min(y_test[:, 0])])
        mask = (t_min < y_test[:, 0]) & (y_test[:, 0] < t_max)
        X_test = X_test[mask]
        y_test = y_test[mask]
        # Create a timeline for evaluation
        t_max = min([t_max, max(y_test[:, 0])])
        t_min = max([t_min, min(y_test[:, 0])])
        timeline = np.linspace(t_min, t_max, timeline_resolution, endpoint=False).astype(np.float32)
        mdn = MDN(input_shape,
                  n_hidden=n_hidden,
                  mixture_components=mixture_components,
                  use_sparse_layer=use_sparse_layer,
                  use_sparse_loss=use_sparse_loss,
                  lmbd=lmbd,
                  use_batchnorm=use_batchnorm,
                  use_dropout=use_dropout,
                  dropout=dropout,
                  mlp_size=(mlp_size_1, mlp_size_2, mlp_size_3),
                  kernel=kernel)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = loss_name(mdn)
        mdn.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
        callbacks = [EarlyStopping(patience=50, verbose=0, restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=0),
                     ModelCheckpoint(f'Logs/model-mdn-{self.dataset_name}.h5', verbose=0, save_best_only=True,
                                     save_weights_only=True)
                     ]
        history = mdn.fit(X_train, y_train,
                          epochs=num_epochs,
                          callbacks=callbacks,
                          validation_data=(X_test, y_test),
                          batch_size=batch_size,
                          verbose=0)

        # Plot loss function
        loss = np.array(history.history["loss"])[~np.isnan(history.history["val_loss"])]
        val_loss = np.array(history.history["val_loss"])[~np.isnan(history.history["val_loss"])]
        plt.figure(figsize=(10, 6))
        plt.title("Learning Curve")
        plt.plot(loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.plot(np.argmin(val_loss), np.min(val_loss), marker="x", color="r", label="Best Model")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend();
        plt.show()

        # Survival function
        mdn.load_weights(f'Logs/model-mdn-{self.dataset_name}.h5')
        os.remove(f'Logs/model-mdn-{self.dataset_name}.h5')
        survs = mdn.predict_survival(X_test, timeline)
        for surv in survs:
            plt.plot(timeline, surv)
        plt.show()
        prob_survival = np.sum(survs, axis=0) / len(X_test)
        km_timeline, km_prob_survival = kaplan_meier_estimator(y_test[:, 1].astype(bool), y_test[:, 0])
        plt.plot(timeline, prob_survival, label=self.model_name)
        plt.plot(km_timeline, km_prob_survival, label="KM")
        plt.legend()
        plt.show()

        # Calculate metrics
        HarrelC = concordance_index_censored_scorer(mdn, timeline, 0, y_test, X_test)
        UnoC = concordance_index_ipcw_scorer(mdn, timeline, 0, y_train, y_test, X_test)
        iBrier = integrated_brier_scorer(mdn, timeline, 0, y_train, y_test, X_test)
        AUROC = cumulative_dynamic_auc_scorer(mdn, timeline, 0, y_train, y_test, X_test)
        LOGR = log_rank_test_scorer(mdn, timeline, 0, y_test, X_test)
        Timeline = timeline.tolist()
        Overall_Survival_Function = prob_survival.tolist()

        #print(HarrelC)
        #print(UnoC)
        #print(iBrier)
        #print(AUROC)
        #print(LOGR)
        return {"HarrelC" : HarrelC,
                "UnoC" : UnoC,
                "iBrier" : iBrier,
                "AUROC" : AUROC,
                "LogRank" : LOGR,
                "Timeline" : Timeline,
                "Overall_Survival_Function" : Overall_Survival_Function}

    def analyze(self,
                n_hidden,
                mixture_components,
                use_sparse_layer,
                use_sparse_loss,
                lmbd,
                use_batchnorm,
                use_dropout,
                dropout,
                mlp_size_1,
                mlp_size_2,
                mlp_size_3,
                kernel,
                loss_name,
                num_epochs,
                batch_size,
                input_shape,
                learning_rate,
                timeline_resolution,
                sensitivity_grid):
        for T_max, red_rate in sensitivity_grid:
            print(f"T_max:{T_max},red_rate:{red_rate}")

            # Open the results table
            if os.path.exists(f"Logs/{self.dataset_name}_results.json"):
                with open(f"Logs/{self.dataset_name}_results.json", "r") as read_file:
                    results_table = json.load(read_file)
            # If it doesn't exist create one
            else:
                results_table = {"Model": [],
                                 "Optimized_LogRank": [],
                                 "T_max": [],
                                 "Uncensored_Reduction": [],
                                 "5Fold": {"HarrelC": {"mean": [],
                                                       "std": []},
                                           "UnoC": {"mean": [],
                                                    "std": []},
                                           "iBrier": {"mean": [],
                                                      "std": []},
                                           "AUROC": {"mean": [],
                                                     "std": []},
                                           "LogRank": {"mean": [],
                                                       "std": []}},
                                 "Hold-out": {"HarrelC": [],
                                              "UnoC": [],
                                              "iBrier": [],
                                              "AUROC": [],
                                              "LogRank": []},
                                 "Timeline": [],
                                 "Overall_Survival_Function": []}
                with open(f"Logs/{self.dataset_name}_results.json", "w") as write_file:
                    json.dump(results_table, write_file)

            # Add properties of experiment
            results_table["Model"].append(self.model_name)
            results_table["Optimized_LogRank"].append(self.Optimized_LogRank)
            results_table["T_max"].append(T_max)
            results_table["Uncensored_Reduction"].append(red_rate)

            # Apply T_max reduction
            y_red = np.copy(self.y)
            y_red = reduce_T_max(y_red, T_max)

            # Split train-holdout sets
            X_train, X_test, y_train, y_test = train_test_split(self.X, y_red, test_size=0.3, random_state=42)

            # 5Fold validation
            scores = self.mdn_kfold_scorer(n_hidden = n_hidden,
                                      mixture_components = mixture_components,
                                      use_sparse_layer = use_sparse_layer,
                                      use_sparse_loss = use_sparse_loss,
                                      lmbd = lmbd,
                                      use_batchnorm = use_batchnorm,
                                      use_dropout = use_dropout,
                                      dropout = dropout,
                                      mlp_size_1 = mlp_size_1,
                                      mlp_size_2 = mlp_size_2,
                                      mlp_size_3 = mlp_size_3,
                                      kernel = kernel,
                                      loss_name = loss_name,
                                      num_epochs = num_epochs,
                                      batch_size = batch_size,
                                      input_shape = input_shape,
                                      learning_rate = learning_rate,
                                      X = X_train,
                                      y = y_train,
                                      red_rate = red_rate,
                                      timeline_resolution = timeline_resolution)
            results_table["5Fold"]["HarrelC"]["mean"].append(scores["HarrelC"]["mean"])
            results_table["5Fold"]["HarrelC"]["std"].append(scores["HarrelC"]["std"])
            results_table["5Fold"]["UnoC"]["mean"].append(scores["UnoC"]["mean"])
            results_table["5Fold"]["UnoC"]["std"].append(scores["UnoC"]["std"])
            results_table["5Fold"]["iBrier"]["mean"].append(scores["iBrier"]["mean"])
            results_table["5Fold"]["iBrier"]["std"].append(scores["iBrier"]["std"])
            results_table["5Fold"]["AUROC"]["mean"].append(scores["AUROC"]["mean"])
            results_table["5Fold"]["AUROC"]["std"].append(scores["AUROC"]["std"])
            results_table["5Fold"]["LogRank"]["mean"].append(scores["LogRank"]["mean"])
            results_table["5Fold"]["LogRank"]["std"].append(scores["LogRank"]["std"])

            # Evaluate model performance on Hold-out set
            results = self.mdn_holdout_scorer(n_hidden = n_hidden,
                                              mixture_components = mixture_components,
                                              use_sparse_layer = use_sparse_layer,
                                              use_sparse_loss = use_sparse_loss,
                                              lmbd = lmbd,
                                              use_batchnorm = use_batchnorm,
                                              use_dropout = use_dropout,
                                              dropout = dropout,
                                              mlp_size_1 = mlp_size_1,
                                              mlp_size_2 = mlp_size_2,
                                              mlp_size_3 = mlp_size_3,
                                              kernel = kernel,
                                              loss_name = loss_name,
                                              num_epochs = num_epochs,
                                              batch_size = batch_size,
                                              input_shape = input_shape,
                                              learning_rate = learning_rate,
                                              X_train = X_train,
                                              X_test = X_test,
                                              y_train = y_train,
                                              y_test = y_test,
                                              red_rate = red_rate,
                                              timeline_resolution = timeline_resolution)
            results_table["Hold-out"]["HarrelC"].append(results["HarrelC"])
            results_table["Hold-out"]["UnoC"].append(results["UnoC"])
            results_table["Hold-out"]["iBrier"].append(results["iBrier"])
            results_table["Hold-out"]["AUROC"].append(results["AUROC"])
            results_table["Hold-out"]["LogRank"].append(results["LogRank"])
            results_table["Timeline"].append(results["Timeline"])
            results_table["Overall_Survival_Function"].append(results["Overall_Survival_Function"])

            # Save results for each set-up
            with open(f"Logs/{self.dataset_name}_results.json", "w") as write_file:
                json.dump(results_table, write_file)
