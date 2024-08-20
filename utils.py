import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.random.set_random_seed(42)

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, BatchNormalization, Dropout
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import mean_squared_error
from sksurv.compare import compare_survival


# Loss functions
class NLLLoss:
    def __init__(self,mdn_model):
        self.model = mdn_model

    def unpack_params(self,params):
        components = self.model.mixture_components
        parameters = self.model.mixture_parameters
        return [params[:,i*components:(i+1)*components] for i in range(parameters)]

    def __call__(self,y,params, **kwargs):
        if y.shape[1] > 1:
            y = tf.transpose(y[:,0])
        else:
            y = y
        alphas, *params = self.unpack_params(params)
        kmm = self.model.kernel_mixture_model(alphas,*params)
        log_likelihood = tf.math.log(tf.add(tf.constant(1e-9, dtype=tf.float32),tf.math.abs(kmm.prob(y))))
        #log_likelihood = kmm.log_prob(y) # Evaluate log-probability of y
        #log_likelihood_without_nans = tf.where(~tf.math.is_finite(log_likelihood), tf.zeros_like(log_likelihood), log_likelihood)
        #tf.print(log_likelihood_without_nans)
        return -tf.reduce_mean(log_likelihood, axis=-1)


class CensoredNLLLoss:
    def __init__(self,mdn_model):
        self.model = mdn_model

    def unpack_params(self,params):
        components = self.model.mixture_components
        parameters = self.model.mixture_parameters
        return [params[:,i*components:(i+1)*components] for i in range(parameters)]

    def unpack_y(self,y):
        if len(np.shape(y)) > 1:
            return [y[:,i] for i in range(2)]
        else:
            return y

    def __call__(self,y,params, **kwargs):
        t, delta = self.unpack_y(y)
        unity = tf.ones_like(delta)
        alphas, *params = self.unpack_params(params)
        kmm = self.model.kernel_mixture_model(alphas,*params)
        #log_likelihood = kmm.log_prob(tf.transpose(t))  # Evaluate log-probability of t
        log_likelihood = tf.math.log(tf.add(tf.constant(1e-9, dtype=tf.float32),tf.math.abs(kmm.prob(tf.transpose(t)))))
        delta_log_likelihood = tf.math.multiply(delta,log_likelihood)
        #log_survival = kmm.log_survival_function(tf.transpose(t)) # Evaluate log-survival of t
        log_survival = tf.math.log(tf.add(tf.constant(1e-9, dtype=tf.float32),tf.math.abs(kmm.survival_function(tf.transpose(t)))))
        delta_log_survival = tf.math.multiply(unity-delta,log_survival)
        #delta_log_likelihood_without_nans = tf.where(~tf.math.is_finite(delta_log_likelihood), tf.zeros_like(delta_log_likelihood), delta_log_likelihood)
        #delta_log_survival_without_nans = tf.where(~tf.math.is_finite(delta_log_survival), tf.zeros_like(delta_log_survival), delta_log_survival)
        #loss = tf.reduce_mean(delta_log_likelihood_without_nans, axis=-1) + tf.reduce_mean(delta_log_survival_without_nans, axis=-1)
        loss = tf.reduce_mean(delta_log_likelihood, axis=-1) + tf.reduce_mean(delta_log_survival, axis=-1)
        # Will have to check how this Loss behaves!!!
        return -loss


class AlternativeNLLLoss:
    def __init__(self,mdn_model):
        self.model = mdn_model

    def unpack_params(self,params):
        components = self.model.mixture_components
        parameters = self.model.mixture_parameters
        return [params[:,i*components:(i+1)*components] for i in range(parameters)]

    def unpack_y(self,y):
        if len(np.shape(y)) > 1:
            return [y[:,i] for i in range(2)]
        else:
            return y
    
    def __call__(self,y,params, **kwargs):
        t, delta = self.unpack_y(y)
        alphas, *params = self.unpack_params(params)
        kmm = self.model.kernel_mixture_model(alphas,*params)
        survival = tf.add(tf.constant(1e-9, dtype=tf.float32),tf.math.abs(kmm.survival_function(tf.transpose(t))))
        log_hazard = tf.math.log(tf.add(tf.constant(1e-9, dtype=tf.float32),tf.math.divide(tf.math.abs(kmm.prob(tf.transpose(t))),survival)))
        delta_log_hazard = tf.math.multiply(delta,log_hazard)
        log_survival = tf.math.log(survival)
        loss = tf.reduce_mean(delta_log_hazard, axis=-1) + tf.reduce_mean(log_survival, axis=-1)
        return -loss


def sparse_loss(w, lmbd = 1e-4):
    return lmbd*tf.reduce_sum(tf.math.sqrt(tf.math.abs(w)))


# Metrics to display during Fit()
# Calling these metrics with .fit results in a significant slowdown!!
class UnoC(tf.keras.metrics.Metric):
    def __init__(self, mdn_model,timeline,y_train, name = 'UnoC', **kwargs):
        super(UnoC, self).__init__(name=name, **kwargs)
        self.model = mdn_model
        self.timeline = timeline
        t, delta = self.unpack_y(y_train)
        self.survival_train = Surv().from_arrays(delta,t)
        self._data = {
            "label_time": [],
            "label_event": [],
            "predicted_params": []
        }

    def reset_state(self):
        self._data = {
            "label_time": [],
            "label_event": [],
            "predicted_params": []
        }

    def unpack_params(self,params):
        components = self.model.mixture_components
        parameters = self.model.mixture_parameters
        return [params[:,i*components:(i+1)*components] for i in range(parameters)]

    def unpack_y(self,y):
        if len(np.shape(y)) > 1:
            return [y[:,i] for i in range(2)]
        else:
            raise TypeError("y needs to contain time and event indicator!")

    def update_state(self, y_true, y_pred, sample_weight=None):
        #print(tf.executing_eagerly())
        self._data["label_time"].append(y_true[:,0].numpy())
        self._data["label_event"].append(y_true[:,1].numpy())
        self._data["predicted_params"].append(y_pred.numpy())

    def result(self):
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)
        t = data["label_time"]
        delta = data["label_event"]
        params = data["predicted_params"]
        survival_test = Surv().from_arrays(delta,t)
        alphas, *params = self.unpack_params(params)
        kmm = self.model.kernel_mixture_model(alphas, *params)
        hazard_funcs = tf.map_fn(tf.function(func = lambda x: -kmm.log_survival_function(x)), tf.constant(self.timeline), parallel_iterations=10).numpy().T # cumulative hazard function = negative log survival
        estimate = []
        for hazard_func in hazard_funcs:
            estimate.append(np.ma.masked_invalid(hazard_func).sum())
        estimate = np.array(estimate)
        return concordance_index_ipcw(self.survival_train,survival_test,estimate)[0]


class IntegratedBrier(tf.keras.metrics.Metric):
    def __init__(self, mdn_model,timeline,y_train, name = 'IntegratedBrier', **kwargs):
        super(IntegratedBrier, self).__init__(name=name, **kwargs)
        self.model = mdn_model
        self.timeline = timeline
        t, delta = self.unpack_y(y_train)
        self.survival_train = Surv().from_arrays(delta,t)
        self._data = {
            "label_time": [],
            "label_event": [],
            "predicted_params": []
        }

    def reset_state(self):
        self._data = {
            "label_time": [],
            "label_event": [],
            "predicted_params": []
        }

    def unpack_params(self,params):
        components = self.model.mixture_components
        parameters = self.model.mixture_parameters
        return [params[:,i*components:(i+1)*components] for i in range(parameters)]

    def unpack_y(self,y):
        if len(np.shape(y)) > 1:
            return [y[:,i] for i in range(2)]
        else:
            raise TypeError("y needs to contain time and event indicator!")

    def update_state(self, y_true, y_pred, sample_weight=None):
        #print(tf.executing_eagerly())
        self._data["label_time"].append(y_true[:,0].numpy())
        self._data["label_event"].append(y_true[:,1].numpy())
        self._data["predicted_params"].append(y_pred.numpy())

    def result(self):
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)
        t = data["label_time"]
        delta = data["label_event"]
        params = data["predicted_params"]
        survival_test = Surv().from_arrays(delta,t)
        alphas, *params = self.unpack_params(params)
        kmm = self.model.kernel_mixture_model(alphas, *params)
        cdfs = tf.map_fn(tf.function(func = lambda x: kmm.cdf(x)), tf.constant(self.timeline), parallel_iterations=10).numpy().T
        return integrated_brier_score(self.survival_train,survival_test,cdfs,self.timeline)


class CumulativeDynamicAUC(tf.keras.metrics.Metric):
    def __init__(self, mdn_model,timeline,y_train, name = 'CumulativeDynamicAUC', **kwargs):
        super(CumulativeDynamicAUC, self).__init__(name=name, **kwargs)
        self.model = mdn_model
        self.timeline = timeline
        t, delta = self.unpack_y(y_train)
        self.survival_train = Surv().from_arrays(delta,t)
        self._data = {
            "label_time": [],
            "label_event": [],
            "predicted_params": []
        }

    def reset_state(self):
        self._data = {
            "label_time": [],
            "label_event": [],
            "predicted_params": []
        }

    def unpack_params(self,params):
        components = self.model.mixture_components
        parameters = self.model.mixture_parameters
        return [params[:,i*components:(i+1)*components] for i in range(parameters)]

    def unpack_y(self,y):
        if len(np.shape(y)) > 1:
            return [y[:,i] for i in range(2)]
        else:
            raise TypeError("y needs to contain time and event indicator!")

    def update_state(self, y_true, y_pred, sample_weight=None):
        #print(tf.executing_eagerly())
        self._data["label_time"].append(y_true[:,0].numpy())
        self._data["label_event"].append(y_true[:,1].numpy())
        self._data["predicted_params"].append(y_pred.numpy())

    def result(self):
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)
        t = data["label_time"]
        delta = data["label_event"]
        params = data["predicted_params"]
        survival_test = Surv().from_arrays(delta,t)
        alphas, *params = self.unpack_params(params)
        kmm = self.model.kernel_mixture_model(alphas, *params)
        cdfs = tf.map_fn(tf.function(func = lambda x: kmm.cdf(x)), tf.constant(self.timeline), parallel_iterations=10).numpy().T
        return np.nanmean(cumulative_dynamic_auc(self.survival_train,survival_test,cdfs,self.timeline)[0])

# Metrics for hyperparameter optimization
def concordance_index_censored_scorer(model, timeline, threshold, y_test, X_test):
    t, delta = y_test[:,0],y_test[:,1].astype(bool)
    hazard_funcs = model.predict_cumulative_hazard(X_test,timeline,threshold)
    estimate = []
    for hazard_func in hazard_funcs:
        estimate.append(np.ma.masked_invalid(hazard_func).sum())
    estimate = np.array(estimate)
    return concordance_index_censored(delta,t,estimate)[0]

def concordance_index_ipcw_scorer(model, timeline, threshold, y_train, y_test, X_test):
    t, delta = y_train[:,0],y_train[:,1].astype(bool)
    survival_train = Surv().from_arrays(delta,t)
    t, delta = y_test[:,0],y_test[:,1].astype(bool)
    survival_test = Surv().from_arrays(delta,t)
    hazard_funcs = model.predict_cumulative_hazard(X_test,timeline,threshold)
    estimate = []
    for hazard_func in hazard_funcs:
        estimate.append(np.ma.masked_invalid(hazard_func).sum())
    estimate = np.array(estimate)
    return concordance_index_ipcw(survival_train, survival_test, estimate)[0]

def integrated_brier_scorer(model, timeline, threshold, y_train, y_test, X_test):
    t, delta = y_train[:,0],y_train[:,1].astype(bool)
    survival_train = Surv().from_arrays(delta,t)
    t, delta = y_test[:,0],y_test[:,1].astype(bool)
    survival_test = Surv().from_arrays(delta,t)
    survivals = model.predict_survival(X_test,timeline,threshold)
    return integrated_brier_score(survival_train,survival_test,survivals,timeline)

def cumulative_dynamic_auc_scorer(model, timeline, threshold, y_train, y_test, X_test):
    t, delta = y_train[:,0],y_train[:,1].astype(bool)
    survival_train = Surv().from_arrays(delta,t)
    t, delta = y_test[:,0],y_test[:,1].astype(bool)
    survival_test = Surv().from_arrays(delta,t)
    hazard_funcs = model.predict_cumulative_hazard(X_test,timeline,threshold)
    return np.nanmean(cumulative_dynamic_auc(survival_train,survival_test,hazard_funcs,timeline)[0])

def root_mean_squared_error_scorer(model, threshold, y_test, X_test, squared = False):
    return mean_squared_error(y_test[:,0],model.predict_mean(X_test,threshold),squared=squared)

def log_rank_test(survs,timeline,y_true):
    survs = np.array(survs)
    norms = (np.ones_like(survs).T*np.sum(np.gradient(1-survs,axis=1),axis = 1)).T
    pdfs = np.gradient(1-survs,axis=1)/norms
    means_pred = np.sum(pdfs*timeline,axis = 1)
    mask = ~np.isnan(means_pred)
    y_true = y_true[mask]
    means_pred = means_pred[mask]
    y_pred = np.stack([means_pred,y_true[:,1]],axis = 1)
    group_indicator = np.concatenate((np.zeros(len(y_true)),np.ones(len(y_true))))
    Y = Surv().from_arrays(np.concatenate((y_true,y_pred))[:,1].astype(bool),np.concatenate((y_true,y_pred))[:,0])
    p_val = compare_survival(Y,group_indicator)[1]
    return p_val

def log_rank_test_scorer(model, timeline, threshold, y_test, X_test):
    survivals = model.predict_survival(X_test,timeline,threshold)
    p_val = log_rank_test(survivals,timeline,y_test)
    return p_val

def reduce_uncensored(y_train,red_rate = 1.0):
    y_red = np.copy(y_train)
    seed = 0
    while np.sum(y_red[:,1] == 1) > np.sum(y_train[:,1] == 1)*red_rate:
        np.random.seed(seed)
        i = np.random.randint(0,len(y_red))
        t_censor = np.random.uniform(0.0,max(y_red[:,0]))
        if (y_red[i,1] == 1) & (y_red[i,0] > t_censor):
            y_red[i,0] = t_censor
            y_red[i,1] = 0
        seed += 1
    return y_red

def reduce_T_max(y,T_max = 1.0):
    y_red = np.copy(y)
    for i in range(len(y_red)):
        if (y_red[i,0] > T_max) & (y_red[i,1] == 1):
            y_red[i,1] = 0
    return y_red