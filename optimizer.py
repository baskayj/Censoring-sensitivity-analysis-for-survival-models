import os
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.random.set_random_seed(42)

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
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
from utils import log_rank_test_scorer, log_rank_test

import optuna

class Optimizer:
    def __init__(self,X,y,
                 name="",
                 timeline_resolution=200,
                 direction="maximize",
                 num_epochs=200,
                 batch_size=128,
                 learning_rate=1e-3,
                 use_kfold=False,
                 use_logrank=False,
                 score="UnoC",
                 kernel_list = ["Exponential","Weibull","Gumbel","Normal","LogNormal","Logistic","LogLogistic","Gamma"]):
        if use_kfold:
            self.X, _ , self.y, _ = train_test_split(X, y, test_size=0.3, random_state=42)
            self.timeline_resolution = timeline_resolution
        else:
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
            t_max = min([max(y_train[:,0]),max(y_test[:,0])])
            t_min = max([min(y_train[:,0]),min(y_test[:,0])])
            mask = (t_min <= y_test[:,0]) & (y_test[:,0] <= t_max)
            X_test = X_test[mask]
            y_test = y_test[mask]
            t_max = min([t_max,max(y_test[:,0])])
            t_min = max([t_min,min(y_test[:,0])])
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.timeline = np.linspace(t_min,t_max,timeline_resolution).astype(np.float32)
        self.name = name
        self.direction = direction
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_kfold = use_kfold
        self.use_logrank = use_logrank
        self.score = score
        self.kernel_list = kernel_list
        
    def objective(self,trial):
        # Hyperparameters
        n_hidden = trial.suggest_categorical("n_hidden",[32,64,128,256,512,1024,2048])
        mixture_components = trial.suggest_int("mixture_components",low=1,high=10)
        use_sparse_layer = trial.suggest_categorical("use_sparse_layer",[True,False])
        if use_sparse_layer:
            use_sparse_loss = trial.suggest_categorical("use_sparse_loss",[True,False])
            if use_sparse_loss:
                lmbd = trial.suggest_float("lmbd",1e-6,1e-1,log=True)
            else:
                lmbd = 0
        else:
            use_sparse_loss = 0
            lmbd = 0
        use_batchnorm = trial.suggest_categorical("use_batchnorm",[True,False])
        use_dropout = trial.suggest_categorical("use_dropout",[True,False])
        if use_dropout:
            dropout = trial.suggest_float("dropout",0,1)
        else:
            dropout = 0
        mlp_size_1 = trial.suggest_int("mlp_size_1",low=0,high=int(np.log2(n_hidden)-2)-1)
        mlp_size_2 = trial.suggest_int("mlp_size_2",low=0,high=int(np.log2(n_hidden)-2)-1-mlp_size_1)
        mlp_size_3 = trial.suggest_int("mlp_size_3",low=0,high=int(np.log2(n_hidden)-2)-1-mlp_size_2-mlp_size_1)
        kernel = trial.suggest_categorical("kernel",self.kernel_list)
        loss_name = trial.suggest_categorical("loss_name",["NLLLoss","CensoredNLLLoss","AlternativeNLLLoss"])
        #threshold = trial.suggest_float("threshold",0,1)

        # KFold Training
        if self.use_kfold:
            kf = KFold(n_splits=5,random_state=42,shuffle=True)
            scores = []
            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                # Filter the test set for t > max(t_train), as these can cause problems for the scores (...)
                t_max = min([max(y_train[:,0]),max(y_test[:,0])])
                t_min = max([min(y_train[:,0]),min(y_test[:,0])])
                mask = (t_min <= y_test[:,0]) & (y_test[:,0] <= t_max)
                X_test = X_test[mask]
                y_test = y_test[mask]
                t_max = min([t_max,max(y_test[:,0])])
                t_min = max([t_min,min(y_test[:,0])])
                timeline = np.linspace(t_min,t_max,self.timeline_resolution).astype(np.float32)
                
                mdn = MDN(np.shape(X_train)[1],n_hidden=n_hidden,mixture_components=mixture_components,
                          use_sparse_layer=use_sparse_layer,use_sparse_loss=use_sparse_loss,lmbd=lmbd,
                          use_batchnorm=use_batchnorm,use_dropout=use_dropout,dropout=dropout,
                          mlp_size=(mlp_size_1,mlp_size_2,mlp_size_3),kernel=kernel)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                if loss_name == "NLLLoss":
                    loss = NLLLoss(mdn)
                elif loss_name == "CensoredNLLLoss":
                    loss = CensoredNLLLoss(mdn)
                else:
                    loss = AlternativeNLLLoss(mdn)
                mdn.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
                callbacks = [EarlyStopping(patience=50,verbose=0,restore_best_weights=True),
                             ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=0),
                             ModelCheckpoint(f"Logs/model-mdn-optuna-trial-{self.name}.h5", verbose=0, save_best_only=True, save_weights_only=True)
                            ]
                history = mdn.fit(X_train,y_train,epochs=self.num_epochs,callbacks=callbacks,validation_data=(X_test,y_test),batch_size=self.batch_size,verbose=0)
                
                # Evaluation
                score = 0
                try:
                    mdn.load_weights(f"Logs/model-mdn-optuna-trial-{self.name}.h5")
                    os.remove(f"Logs/model-mdn-optuna-trial-{self.name}.h5")
                    if self.use_logrank:
                        try:
                            p_val = log_rank_test_scorer(model = mdn, timeline = timeline, threshold = 0, y_test = y_test, X_test = X_test)
                            # It only matters, if it passes the test, we don't care by how much...
                            if p_val > 0.05:
                                score += 1
                            else:
                                score += 0
                        except ValueError:
                            score += -1
                    try:
                        if self.score == "UnoC":
                            score += concordance_index_ipcw_scorer(model = mdn, timeline = timeline, threshold = 0, y_train = y_train, y_test = y_test, X_test = X_test)
                        elif self.score == "HarrellC":
                            score += concordance_index_censored_scorer(model = mdn, timeline = timeline, threshold = 0, y_test = y_test, X_test = X_test)
                        elif self.score == "ibrier":
                            score += integrated_brier_scorer(model = mdn, timeline = timeline, threshold = 0, y_train = y_train, y_test = y_test, X_test = X_test)
                        elif self.score == "auroc":
                            score += cumulative_dynamic_auc_scorer(model = mdn, timeline = timeline, threshold = 0, y_train = y_train, y_test = y_test, X_test = X_test)
                        elif self.score == "rmse":
                            score += root_mean_squared_error_scorer(model = mdn, threshold = 0, y_test = y_test, X_test = X_test)
                        else:
                            raise NameError("Unknown score! Please choose one of the following instead: [Default]UnoC, HarrellC, ibrier, auroc, rmse")
                    except ValueError:
                        score += -1
                except ValueError:
                    score += -1
                except FileNotFoundError:
                    score += -1
                if self.use_logrank:
                    scores.append(score/2)
                else:
                    scores.append(score)
            scores = np.array(scores)
            return np.mean(scores)     
                
        # Normal Training
        else:
            mdn = MDN(np.shape(self.X_train)[1],n_hidden=n_hidden,mixture_components=mixture_components,
                      use_sparse_layer=use_sparse_layer,use_sparse_loss=use_sparse_loss,lmbd=lmbd,
                      use_batchnorm=use_batchnorm,use_dropout=use_dropout,dropout=dropout,
                      mlp_size=(mlp_size_1,mlp_size_2,mlp_size_3),kernel=kernel)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            if loss_name == "NLLLoss":
                loss = NLLLoss(mdn)
            elif loss_name == "CensoredNLLLoss":
                loss = CensoredNLLLoss(mdn)
            else:
                loss = AlternativeNLLLoss(mdn)

            mdn.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
            callbacks = [EarlyStopping(patience=50,verbose=0,restore_best_weights=True),
                         ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=0),
                         ModelCheckpoint(f"Logs/model-mdn-optuna-trial-{self.name}.h5", verbose=0, save_best_only=True, save_weights_only=True)
                        ]
            history = mdn.fit(self.X_train,self.y_train,epochs=self.num_epochs,callbacks=callbacks,validation_data=(self.X_test,self.y_test),batch_size=self.batch_size,verbose=0)

            # Evaluation
            score = 0
            try:
                mdn.load_weights(f"Logs/model-mdn-optuna-trial-{self.name}.h5")
                os.remove(f"Logs/model-mdn-optuna-trial-{self.name}.h5")
                if self.use_logrank:
                    try:
                        p_val = log_rank_test_scorer(model = mdn, timeline = self.timeline, threshold = 0, y_test = self.y_train, X_test = self.X_test)
                        # It only matters, if it passes the test, we don't care by how much...
                        if p_val > 0.05:
                            score += 1
                        else:
                            score += 0
                    except ValueError:
                        score += -1
                try:
                    if self.score == "UnoC":
                        score += concordance_index_ipcw_scorer(model = mdn, timeline = self.timeline, threshold = 0, y_train = self.y_train, y_test = self.y_test, X_test = self.X_test)
                    elif self.score == "HarrellC":
                        score += concordance_index_censored_scorer(model = mdn, timeline = self.timeline, threshold = 0, y_test = self.y_test, X_test = self.X_test)
                    elif self.score == "ibrier":
                        score += integrated_brier_scorer(model = mdn, timeline = self.timeline, threshold = 0, y_train = self.y_train, y_test = self.y_test, X_test = self.X_test)
                    elif self.score == "auroc":
                        score += cumulative_dynamic_auc_scorer(model = mdn, timeline = self.timeline, threshold = 0, y_train = self.y_train, y_test = self.y_test, X_test = self.X_test)
                    elif self.score == "rmse":
                        score += root_mean_squared_error_scorer(model = mdn, threshold = 0, y_test = self.y_test, X_test = self.X_test)
                    else:
                        raise NameError("Unknown score! Please choose one of the following instead: [Default]UnoC, HarrellC, ibrier, auroc, rmse")
                except ValueError:
                    score += -1
            except ValueError:
                score += -1
            except FileNotFoundError:
                score += -1
            if self.use_logrank:
                return score/2
            else:
                return score
    
    def __call__(self,n_trials=100):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_value,study.best_params
    
    
class ML_Optimizer:
    def __init__(self,X,y,
                 model="CoxPH",
                 name="",
                 direction="maximize",
                 use_kfold=False,
                 use_logrank=False,
                 score="UnoC"):
        if use_kfold:
            self.X, _, self.y, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
            t_max = min([max(y_train[:,0]),max(y_test[:,0])])
            t_min = max([min(y_train[:,0]),min(y_test[:,0])])
            mask = (t_min <= y_test[:,0]) & (y_test[:,0] <= t_max)
            X_test = X_test[mask]
            y_test = y_test[mask]
            self.t_max = min([t_max,max(y_test[:,0])])
            self.t_min = max([t_min,min(y_test[:,0])])
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        self.model_name = model
        self.name = name
        self.direction = direction
        self.use_kfold = use_kfold
        self.use_logrank = use_logrank
        self.score = score
        
    def get_score(self,Model,**kwargs):
        if self.use_kfold:
            kf = KFold(n_splits=5,random_state=42,shuffle=True)
            scores = []
            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                # Filter the test set for t > max(t_train), as these can cause problems for the scores (...)
                t_max = min([max(y_train[:,0]),max(y_test[:,0])])
                t_min = max([min(y_train[:,0]),min(y_test[:,0])])
                mask = (t_min <= y_test[:,0]) & (y_test[:,0] <= t_max)
                X_test = X_test[mask]
                y_test = y_test[mask]
                t_max = min([t_max,max(y_test[:,0])])
                t_min = max([t_min,min(y_test[:,0])])
                model = Model(**kwargs)
                t, delta = y_train[:,0],y_train[:,1].astype(bool)
                survival_train = Surv().from_arrays(delta,t)
                t, delta = y_test[:,0],y_test[:,1].astype(bool)
                survival_test = Surv().from_arrays(delta,t)
                score = 0
                try:
                    model.fit(X_train,survival_train)
                    if self.use_logrank:
                        try:
                            survivals = model.predict_survival_function(X_test)
                            mask = (t_min <= survivals[0].x) & (survivals[0].x <= t_max)
                            timeline = survivals[0].x[mask][:-1]
                            survs = []
                            for survival in survivals:
                                    survs.append(survival.y[mask][:-1])
                            p_val = log_rank_test(survs,timeline,y_test)
                            # It only matters, if it passes the test, we don't care by how much...
                            if p_val > 0.05:
                                score += 1
                            else:
                                score += 0
                        except ValueError:
                            score += -1
                    try:
                        if self.score == "UnoC":
                            score += concordance_index_ipcw(survival_train, survival_test, model.predict(X_test))[0]
                        elif self.score == "HarrellC":
                            score += concordance_index_censored(delta.astype(bool), t, model.predict(X_test))[0]
                        elif self.score == "ibrier":
                            try:
                                survivals = model.predict_survival_function(X_test)
                                mask = (t_min <= survivals[0].x) & (survivals[0].x <= t_max)
                                timeline = survivals[0].x[mask][:-1]
                                survs = []
                                for survival in survivals:
                                    survs.append(survival.y[mask][:-1])
                                score += integrated_brier_score(survival_train,survival_test,survs,timeline)
                            except AttributeError:
                                score += -1
                        elif self.score == "auroc":
                            try:
                                hazards = model.predict_cumulative_hazard_function(X_test)
                                mask = (t_min <= survivals[0].x) & (survivals[0].x <= t_max)
                                timeline = hazards[0].x[mask][:-1]
                                hazs = []
                                for hazard in hazards:
                                    hazs.append(hazard.y[mask][:-1])
                                score += np.nanmean(cumulative_dynamic_auc(survival_train,survival_test,hazs,timeline)[0])
                            except AttributeError:
                                score += -1
                        else:
                            raise NameError("Unknown score! Please choose one of the following instead: [Default]UnoC, HarrellC, ibrier, auroc")
                    except ValueError:
                        score += -1
                except ValueError:
                    score += -1
                if self.use_logrank:
                    scores.append(score/2)
                else:
                    scores.append(score)
            scores = np.array(scores)
            return np.mean(scores)
   
        else:
            model = Model(**kwargs)
            t, delta = self.y_train[:,0],self.y_train[:,1].astype(bool)
            survival_train = Surv().from_arrays(delta,t)
            t, delta = self.y_test[:,0],self.y_test[:,1].astype(bool)
            survival_test = Surv().from_arrays(delta,t)
            score = 0
            try:
                model.fit(self.X_train,survival_train)
                if self.use_logrank:
                    try:
                        survivals = model.predict_survival_function(self.X_test)
                        mask = (self.t_min <= survivals[0].x) & (survivals[0].x <= self.t_max)
                        timeline = survivals[0].x[mask][:-1]
                        survs = []
                        for survival in survivals:
                            survs.append(survival.y[mask][:-1])
                        p_val = log_rank_test(survs,timeline,self.y_test)
                        # It only matters, if it passes the test, we don't care by how much...
                        if p_val > 0.05:
                            score += 1
                        else:
                            score += 0
                    except ValueError:
                        score += -1
                try:
                    if self.score == "UnoC":
                        score += concordance_index_ipcw(survival_train, survival_test, model.predict(self.X_test))[0]
                    elif self.score == "HarrellC":
                        score += concordance_index_censored(delta.astype(bool), t, model.predict(self.X_test))[0]
                    elif self.score == "ibrier":
                        try:
                            survivals = model.predict_survival_function(self.X_test)
                            mask = (self.t_min <= survivals[0].x) & (survivals[0].x <= self.t_max)
                            timeline = survivals[0].x[mask][:-1]
                            survs = []
                            for survival in survivals:
                                survs.append(survival.y[mask][:-1])
                            score += integrated_brier_score(survival_train,survival_test,survs,timeline)
                        except AttributeError:
                            score += -1
                    elif self.score == "auroc":
                        try:
                            hazards = model.predict_cumulative_hazard_function(self.X_test)
                            mask = (self.t_min <= survivals[0].x) & (survivals[0].x <= self.t_max)
                            timeline = hazards[0].x[mask][:-1]
                            hazs = []
                            for hazard in hazards:
                                hazs.append(hazard.y[mask][:-1])
                            score += np.nanmean(cumulative_dynamic_auc(survival_train,survival_test,hazs,timeline)[0])
                        except AttributeError:
                            score += -1
                    else:
                        raise NameError("Unknown score! Please choose one of the following instead: [Default]UnoC, HarrellC, ibrier, auroc")
                except ValueError:
                    score += -1
            except ValueError:
                score += -1
            if self.use_logrank:
                return score/2
            else:
                return score
        
    def objective(self,trial):
        if self.model_name == "CoxPH":
            alpha = trial.suggest_float("alpha",low=1e-3,high=5)
            ties = trial.suggest_categorical("ties",["breslow","efron"])
            return self.get_score(CoxPHSurvivalAnalysis,alpha=alpha,ties=ties,n_iter=2000)
        elif self.model_name == "SurvivalSVM":
            alpha = trial.suggest_float("alpha",low=1e-3,high=5)
            rank_ratio  = trial.suggest_float("rank_ratio ",low=0,high=0.99,step=0.01)
            fit_intercept = trial.suggest_categorical("fit_intercept",[True,False])
            kernel = trial.suggest_categorical("kernel",["linear","poly","rbf","sigmoid","cosine"])
            degree = trial.suggest_int("degree",low=0,high=5)
            coef0 = trial.suggest_float("coef0",low=0,high=1)
            optimizer = trial.suggest_categorical("optimizer",["avltree","rbtree"])
            return self.get_score(FastKernelSurvivalSVM,alpha=alpha,rank_ratio=rank_ratio,fit_intercept=fit_intercept,kernel=kernel,degree=degree,coef0=coef0,max_iter=200,optimizer=optimizer)
        elif self.model_name == "SurvivalTree":
            splitter = trial.suggest_categorical("splitter",["best","random"])
            min_samples_split = trial.suggest_float("min_samples_split",low=1e-3,high=0.5)
            min_samples_leaf = trial.suggest_float("min_samples_leaf",low=1e-3,high=0.5)
            min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf",low=0,high=0.5)
            max_features = trial.suggest_float("max_features",low=0,high=1)
            return self.get_score(SurvivalTree,splitter=splitter,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,random_state=42)
        elif self.model_name == "RandomSurvivalForest":
            n_estimators = trial.suggest_int("n_estimators",low=100,high=1000,log=True)
            min_samples_split = trial.suggest_float("min_samples_split",low=1e-3,high=0.5)
            min_samples_leaf = trial.suggest_float("min_samples_leaf",low=1e-3,high=0.5)
            min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf",low=0,high=0.5)
            max_features = trial.suggest_float("max_features",low=0,high=1)
            bootstrap = trial.suggest_categorical("bootstrap",[True,False])
            max_samples = trial.suggest_float("max_samples",low=0,high=1)
            if bootstrap == True:
                oob_score = trial.suggest_categorical("oob_score",[True,False])
            else:
                oob_score = False
            return self.get_score(RandomSurvivalForest,n_estimators=n_estimators,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,bootstrap=bootstrap,oob_score=oob_score,random_state=42,n_jobs=10,max_samples=max_samples)
        elif self.model_name == "BoostingSurvival":
            loss = trial.suggest_categorical("loss",["coxph","squared"])
            learning_rate = trial.suggest_float("learning_rate",low=1e-6,high=1e-1,log=True)
            n_estimators = trial.suggest_int("n_estimators",low=100,high=1000,log=True)
            criterion = trial.suggest_categorical("criterion",["friedman_mse","squared_error","absolute_error"])
            min_samples_split = trial.suggest_int("min_samples_split",low=2,high=10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf",low=1,high=10)
            min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf",low=1e-3,high=0.5)
            min_impurity_decrease = trial.suggest_float("min_impurity_decrease",low=1e-3,high=1)
            max_features = trial.suggest_float("max_features",low=1e-3,high=1)
            subsample = trial.suggest_float("subsample",low=1e-3,high=1)
            dropout_rate = trial.suggest_float("dropout_rate",low=0,high=1)
            ccp_alpha = trial.suggest_float("ccp_alpha",low=1e-3,high=5)
            return self.get_score(GradientBoostingSurvivalAnalysis,loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,criterion=criterion,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,min_impurity_decrease=min_impurity_decrease,max_features=max_features,subsample=subsample,dropout_rate=dropout_rate,ccp_alpha=ccp_alpha,random_state=42)
        else:
            raise NameError("Unknown model! Please choose one of the following: [Default]CoxPH, SurvivalSVM, SurvivalTree, RandomSurvivalForest, BoostingSurvival")
            
    def __call__(self,n_trials=100):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_value,study.best_params