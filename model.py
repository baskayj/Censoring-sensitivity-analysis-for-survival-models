import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.random.set_random_seed(42)

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, BatchNormalization, Dropout
from utils import sparse_loss


def nnelu(x):
    #Computes the Non-Negative Exponential Linear Unit
    #https://gist.github.com/oborchers/2732decb2b1cfd878ce14df0bfd76a30
    return tf.add(tf.constant(1+1e-9, dtype=tf.float32), tf.nn.elu(x))
tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

class SparseMixture(tf.keras.layers.Layer):
    def __init__(self,mixture_components,use_sparse_loss = False, lmbd = 1e-4):
        super(SparseMixture, self).__init__(name="SparseMixture")
        self.mixture_components = mixture_components
        self.use_sparse_loss = use_sparse_loss
        self.lmbd = lmbd
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(self.mixture_components,), dtype="float32"),trainable=True,)

    def call(self,alphas):
        w = tf.math.abs(self.w)/tf.reshape(tf.reduce_sum(tf.math.abs(self.w),axis=-1),(-1,1)) # Normalize weights
        new_alphas = tf.math.multiply(w,alphas) # Multiply mixing components with the normalized weights
        new_alphas = new_alphas/tf.reshape(tf.reduce_sum(new_alphas,axis=-1),(-1,1)) # Normalize mixing components
        if self.use_sparse_loss:
            self.add_loss(sparse_loss(self.w, self.lmbd))
        return new_alphas


class MDN(tf.keras.Model):
    def __init__(self, x_shape, n_hidden, mixture_components, use_sparse_layer = True, use_sparse_loss = False, lmbd = 1e-4,use_batchnorm = False, use_dropout = False, dropout = 0.1 , mlp_size = (1,0,0), kernel = "Exponential"):
        super(MDN, self).__init__(name="MDN")
        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.mixture_components = mixture_components
        self.use_sparse_layer = use_sparse_layer
        self.use_sparse_loss = use_sparse_loss
        self.lmbd = lmbd
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout = dropout
        if ((mlp_size[0]+mlp_size[1])>=(np.log2(self.n_hidden)-2))or((mlp_size[0]+mlp_size[2])>=(np.log2(self.n_hidden)-2)):
            raise ValueError(f"Number of neurons too small!")
        else:
            self.mlp_size = mlp_size[0]
            self.alpha_mlp_size = mlp_size[1]
            self.kernel_mlp_size = mlp_size[2]
        self.kernel_name = kernel
        # Kernel specific part
        if self.kernel_name == "Exponential":
            # Parameters: [0]=rate
            self.mixture_parameters = 2  # Should be the number of kernel params + 1 (which is alpha)
        elif self.kernel_name == "Weibull":
            # Parameters: [0]=shape/concentration >= 1 to ensure the convexity of the loss function, [1]=scale
            self.mixture_parameters = 3  # Should be the number of kernel params + 1 (which is alpha)
        elif self.kernel_name == "Gumbel":
            # Parameters: [0]=loc,[1]=scale
            self.mixture_parameters = 3  # Should be the number of kernel params + 1 (which is alpha)        
        elif self.kernel_name == "Normal":
            # Parameters: [0]=mu,[1]=sigma
            self.mixture_parameters = 3 # Should be the number of kernel params + 1 (which is alpha)
        elif self.kernel_name == "LogNormal":
            # Parameters: [0]=loc/mu,[1]=scale/sigma
            self.mixture_parameters = 3  # Should be the number of kernel params + 1 (which is alpha)
        elif self.kernel_name == "Logistic":
            # Parameters: [0]=mu/loc,[1]=scale
            self.mixture_parameters = 3  # Should be the number of kernel params + 1 (which is alpha)
        elif self.kernel_name == "LogLogistic":
            # Parameters: [0]=loc,[1]=scale
            self.mixture_parameters = 3  # Should be the number of kernel params + 1 (which is alpha)
        elif self.kernel_name == "Gamma":
            # Parameters: [0]=concentration,[1]=rate
            self.mixture_parameters = 3  # Should be the number of kernel params + 1 (which is alpha)
        else:
            raise NameError("Unknown kernel! Please choose one of the following instead: [Default]Exponential, Weibull, Gumbel, Normal, LogNormal, Logistic, LogLogistic, Gamma")

        self.mlp = tf.keras.Sequential(
            [Input(shape = (self.x_shape,),name="input"),
            Dense(self.n_hidden,input_shape=(self.x_shape,), activation=None, use_bias=True, name="h1")]
            +
            [[BatchNormalization(name="bn1")] if (self.use_batchnorm and self.mlp_size == 0 ) else []][0]
            +
            [Activation("relu",name="a1")]
            +
            [[Dropout(rate=self.dropout,name="do1")] if (self.use_dropout and self.mlp_size == 0 ) else []][0]
            +
            [Dense(int(self.n_hidden/(2**(i+1))),input_shape=(int(self.n_hidden/(2**i)),), activation="relu", use_bias=True, name=f"h{i+2}") for i in range(self.mlp_size-1)]
            +
            [[Dense(int(self.n_hidden/(2**(self.mlp_size))),input_shape=(int(self.n_hidden/(2**(self.mlp_size-1))),), activation=None, use_bias=True, name=f"h{self.mlp_size + 1}")] if self.mlp_size > 0 else []][0]
            +
            [[BatchNormalization(name="bn1")] if (self.use_batchnorm and self.mlp_size > 0 ) else []][0]
            +
            [[Activation("relu",name="a2")] if self.mlp_size > 0 else []][0]
            +
            [[Dropout(rate=self.dropout,name="do1")] if (self.use_dropout and self.mlp_size > 0 ) else []][0]
            )
        # MLP for the mixing coefficients
        self.alpha_mlp = tf.keras.Sequential(
            [[Dense(int(self.n_hidden/(2**(self.mlp_size+1))),input_shape=(int(self.n_hidden/(2**(self.mlp_size))),), activation=None, use_bias=True, name=f"h{self.mlp_size + 2}")] if self.alpha_mlp_size > 0 else []][0]
            +
            [[BatchNormalization(name="bn2")] if (self.use_batchnorm and self.alpha_mlp_size == 1 ) else []][0]
            +
            [[Activation("relu",name="a3")] if self.alpha_mlp_size == 1 else []][0]
            +
            [[Dropout(rate=self.dropout,name="do2")] if (self.use_dropout and self.alpha_mlp_size == 1 ) else []][0]
            +
            [Dense(int(self.n_hidden/(2**(self.mlp_size + 2 + i))), input_shape=(int(self.n_hidden/(2**(self.mlp_size + 1 + i))),), activation="relu", use_bias=True, name=f"h{i + 3 + self.mlp_size}") for i in range(self.alpha_mlp_size-2)]
            +
            [[Dense(int(self.n_hidden/(2**(self.mlp_size+self.alpha_mlp_size))),input_shape=(int(self.n_hidden/(2**(self.mlp_size+self.alpha_mlp_size-1))),), activation=None, use_bias=True, name=f"h{self.mlp_size + self.alpha_mlp_size + 1}")] if self.alpha_mlp_size > 1 else []][0]
            +
            [[BatchNormalization(name="b2")] if (self.use_batchnorm and self.alpha_mlp_size > 1 ) else []][0]
            +
            [[Activation("relu",name="a4")] if self.alpha_mlp_size > 1 else []][0]
            +
            [[Dropout(rate=self.dropout,name="d2")] if (self.use_dropout and self.alpha_mlp_size > 1 ) else []][0]
            )
        self.alpha_layer = Dense(self.mixture_components,activation="softmax", use_bias=False, name="alpha_layer")
        if self.use_sparse_layer:
            self.sparse_mixture_layer = SparseMixture(self.mixture_components,use_sparse_loss=self.use_sparse_loss,lmbd=self.lmbd)
        # MLP for the kernel parameters
        self.kernel_mlp = tf.keras.Sequential(
            [[Dense(int(self.n_hidden/(2**(self.mlp_size+1))),input_shape=(int(self.n_hidden/(2**(self.mlp_size))),), activation=None, use_bias=True, name=f"h{self.mlp_size + self.alpha_mlp_size + 2}")] if self.kernel_mlp_size > 0 else []][0]
            +
            [[BatchNormalization(name="bn3")] if (self.use_batchnorm and self.kernel_mlp_size == 1 ) else []][0]
            +
            [[Activation("relu",name="a5")] if self.kernel_mlp_size == 1 else []][0]
            +
            [[Dropout(rate=self.dropout,name="do3")] if (self.use_dropout and self.kernel_mlp_size == 1 ) else []][0]
            +
            [Dense(int(self.n_hidden/(2**(self.mlp_size + 2 + i))), input_shape=(int(self.n_hidden/(2**(self.mlp_size + 1 + i))),), activation="relu", use_bias=True, name=f"h{i + 3 + self.mlp_size + + self.alpha_mlp_size}") for i in range(self.kernel_mlp_size-2)]
            +
            [[Dense(int(self.n_hidden/(2**(self.mlp_size+self.kernel_mlp_size))),input_shape=(int(self.n_hidden/(2**(self.mlp_size+self.kernel_mlp_size-1))),), activation=None, use_bias=True, name=f"h{self.mlp_size + self.alpha_mlp_size + self.kernel_mlp_size + 1}")] if self.kernel_mlp_size > 1 else []][0]
            +
            [[BatchNormalization(name="b3")] if (self.use_batchnorm and self.kernel_mlp_size > 1 ) else []][0]
            +
            [[Activation("relu",name="a6")] if self.kernel_mlp_size > 1 else []][0]
            +
            [[Dropout(rate=self.dropout,name="d3")] if (self.use_dropout and self.kernel_mlp_size > 1 ) else []][0]
            )

        # Kernel specific part
        if self.kernel_name == "Exponential":
            self.rate_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="rate_layer")
        elif self.kernel_name == "Weibull":
            self.shape_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="shape_layer")
            self.scale_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="scale_layer")
        elif self.kernel_name == "Gumbel":
            self.loc_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="loc_layer")
            self.scale_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="scale_layer")
        elif self.kernel_name == "Normal":
            self.mu_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="mu_layer")
            self.sigma_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="sigma_layer")
        elif self.kernel_name == "LogNormal":
            self.mu_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="mu_layer")
            self.sigma_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="sigma_layer")
        elif self.kernel_name == "Logistic":
            self.mu_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="mu_layer")
            self.scale_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="scale_layer")
        elif self.kernel_name == "LogLogistic":
            self.loc_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="loc_layer")
            self.scale_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="scale_layer")
        elif self.kernel_name == "Gamma":
            self.con_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="con_layer")
            self.rate_layer = Dense(self.mixture_components, activation="nnelu", use_bias=False, name="rate_layer")

    def encode(self,inputs):
        z = self.mlp(inputs)
        z_alphas = self.alpha_mlp(z)
        alphas = self.alpha_layer(z_alphas)
        if self.use_sparse_layer:
            alphas = self.sparse_mixture_layer(alphas)
        z_kernel = self.kernel_mlp(z)
        # Kernel specific part
        if self.kernel_name == "Exponential":
            rates = self.rate_layer(z_kernel)
            return alphas,rates
        elif self.kernel_name == "Weibull":
            shapes = self.shape_layer(z_kernel)
            shapes = tf.add(tf.constant(1, dtype=tf.float32), shapes) # Offsetting the shape parameter to be at least 1
            scales = self.scale_layer(z_kernel)
            return alphas,shapes,scales
        elif self.kernel_name == "Gumbel":
            locs = self.loc_layer(z_kernel)
            scales = self.scale_layer(z_kernel)
            return alphas,locs,scales
        elif self.kernel_name == "Normal":
            mus = self.mu_layer(z_kernel)
            sigmas = self.sigma_layer(z_kernel)
            return alphas,mus,sigmas
        elif self.kernel_name == "LogNormal":
            mus = self.mu_layer(z_kernel)
            sigmas = self.sigma_layer(z_kernel)
            return alphas,mus,sigmas
        elif self.kernel_name == "Logistic":
            mus = self.mu_layer(z_kernel)
            scales = self.scale_layer(z_kernel)
            return alphas,mus,scales
        elif self.kernel_name == "LogLogistic":
            mus = self.loc_layer(z_kernel)
            scales = self.scale_layer(z_kernel)
            return alphas,mus,scales
        elif self.kernel_name == "Gamma":
            cons = self.con_layer(z_kernel)
            rates = self.rate_layer(z_kernel)
            return alphas,cons,rates


    def kernel_mixture_model(self,alphas,*params):
        # Kernel specific part
        if self.kernel_name == "Exponential":
            kernel = tfd.Exponential(rate=params[0])
        elif self.kernel_name == "Weibull":
            kernel = tfd.Weibull(concentration=params[0],scale=params[1])
        elif self.kernel_name == "Gumbel":
            kernel = tfd.Gumbel(loc=params[0],scale=params[1])            
        elif self.kernel_name == "Normal":
            kernel = tfd.Normal(loc=params[0],scale=params[1])
        elif self.kernel_name == "LogNormal":
            kernel = tfd.LogNormal(loc=params[0],scale=params[1])
        elif self.kernel_name == "Logistic":
            kernel = tfd.Logistic(loc=params[0],scale=params[1])
        elif self.kernel_name == "LogLogistic":
            kernel = tfd.Logistic(loc=params[0],scale=params[1])
        elif self.kernel_name == "Gamma":
            kernel = tfd.Gamma(concentration=params[0],rate=params[1])
        # Making the Mixture
        kmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alphas),
            components_distribution=kernel
        )
        return kmm

    def predict_cdf(self,inputs,timeline,threshold=0):
        alphas, *params = self.encode(inputs)
        #Threshold the results for smoother CDF
        params = list(map(lambda x:tf.math.multiply(x, tf.cast(alphas>threshold,x.dtype)),params))
        alphas = tf.math.multiply(alphas, tf.cast(alphas>threshold,alphas.dtype))
        alphas = alphas/tf.reshape(tf.reduce_sum(alphas,axis=-1),(-1,1))
        #Build the model with thresholded values, and predict
        kmm = self.kernel_mixture_model(alphas,*params)
        cdfs = tf.map_fn(tf.function(func = lambda x: kmm.cdf(x)), tf.constant(timeline), parallel_iterations=10).numpy().T
        cdfs = np.array(cdfs)
        return cdfs
    
    def predict_pdf(self,inputs,timeline,threshold=0):
        alphas, *params = self.encode(inputs)
        #Threshold the results for smoother CDF
        params = list(map(lambda x:tf.math.multiply(x, tf.cast(alphas>threshold,x.dtype)),params))
        alphas = tf.math.multiply(alphas, tf.cast(alphas>threshold,alphas.dtype))
        alphas = alphas/tf.reshape(tf.reduce_sum(alphas,axis=-1),(-1,1))
        #Build the model with thresholded values, and predict
        kmm = self.kernel_mixture_model(alphas,*params)  
        cdfs = tf.map_fn(tf.function(func = lambda x: kmm.cdf(x)), tf.constant(timeline), parallel_iterations=10).numpy().T
        norms = (np.ones_like(cdfs).T*np.sum(np.gradient(cdfs,axis=1),axis = 1)).T
        pdfs = np.gradient(cdfs,axis=1)/norms
        #pdfs = tf.map_fn(tf.function(func = lambda x: kmm.prob(x)), tf.constant(timeline), parallel_iterations=10).numpy().T
        #pdfs = np.array(pdfs)
        return pdfs
        
    def predict_survival(self,inputs,timeline,threshold=0):
        alphas, *params = self.encode(inputs)
        #Threshold the results for smoother CDF
        params = list(map(lambda x:tf.math.multiply(x, tf.cast(alphas>threshold,x.dtype)),params))
        alphas = tf.math.multiply(alphas, tf.cast(alphas>threshold,alphas.dtype))
        alphas = alphas/tf.reshape(tf.reduce_sum(alphas,axis=-1),(-1,1))
        #Build the model with thresholded values, and predict
        kmm = self.kernel_mixture_model(alphas,*params)  
        survs = tf.map_fn(tf.function(func = lambda x: kmm.survival_function(x)), tf.constant(timeline), parallel_iterations=10).numpy().T
        survs = np.array(survs)
        return survs

    def predict_hazard(self,inputs,timeline,threshold=0):
        alphas, *params = self.encode(inputs)
        #Threshold the results for smoother CDF
        params = list(map(lambda x:tf.math.multiply(x, tf.cast(alphas>threshold,x.dtype)),params))
        alphas = tf.math.multiply(alphas, tf.cast(alphas>threshold,alphas.dtype))
        alphas = alphas/tf.reshape(tf.reduce_sum(alphas,axis=-1),(-1,1))
        #Build the model with thresholded values, and predict
        kmm = self.kernel_mixture_model(alphas,*params)
        cdfs = tf.map_fn(tf.function(func = lambda x: kmm.cdf(x)), tf.constant(timeline), parallel_iterations=10).numpy().T
        survs = tf.map_fn(tf.function(func = lambda x: kmm.survival_function(x)), tf.constant(timeline), parallel_iterations=10).numpy().T
        norms = (np.ones_like(cdfs).T*np.sum(np.gradient(cdfs,axis=1),axis = 1)).T
        pdfs = np.gradient(cdfs,axis=1)/norms
        hazards = pdfs/survs
        #hazards = tf.map_fn(tf.function(func = lambda x: tf.math.divide(kmm.prob(x),tf.add(tf.constant(1e-9, dtype=tf.float32),kmm.survival_function(x)))), tf.constant(timeline), parallel_iterations=10).numpy().T
        #hazards = np.array(hazards)
        return hazards

    def predict_cumulative_hazard(self,inputs,timeline,threshold=0):
        alphas, *params = self.encode(inputs)
        #Threshold the results for smoother CDF
        params = list(map(lambda x:tf.math.multiply(x, tf.cast(alphas>threshold,x.dtype)),params))
        alphas = tf.math.multiply(alphas, tf.cast(alphas>threshold,alphas.dtype))
        alphas = alphas/tf.reshape(tf.reduce_sum(alphas,axis=-1),(-1,1))
        #Build the model with thresholded values, and predict
        kmm = self.kernel_mixture_model(alphas,*params)
        cum_hazards = tf.map_fn(tf.function(func = lambda x: -tf.math.log(tf.add(tf.constant(1e-9, dtype=tf.float32),kmm.survival_function(x)))), tf.constant(timeline), parallel_iterations=10).numpy().T
        cum_hazards = np.array(cum_hazards)
        return cum_hazards
        
    def predict_mean(self,inputs,timeline,threshold=0,predict_stddev=False):
        alphas, *params = self.encode(inputs)
        # Threshold the results
        params = list(map(lambda x:tf.math.multiply(x, tf.cast(alphas>threshold,x.dtype)),params))
        alphas = tf.math.multiply(alphas, tf.cast(alphas>threshold,alphas.dtype))
        alphas = alphas/tf.reshape(tf.reduce_sum(alphas,axis=-1),(-1,1))
        # Build the model with thresholded values, and predict
        kmm = self.kernel_mixture_model(alphas,*params)
        cdfs = tf.map_fn(tf.function(func = lambda x: kmm.cdf(x)), tf.constant(timeline), parallel_iterations=10).numpy().T
        cdfs = np.array(cdfs)
        norms = (np.ones_like(cdfs).T*np.sum(np.gradient(cdfs,axis=1),axis = 1)).T
        pdfs = np.gradient(cdfs,axis=1)/norms
        means = np.sum(pdfs*timeline,axis = 1)
        if not predict_stddev:
            return means
        else:
            stddevs = np.sqrt(np.sum(pdfs*np.square(timeline-means),axis = 1))
            return means,stddevs

    def call(self, inputs, training=None, mask=None):
        alphas, *params = self.encode(inputs)
        return Concatenate()([alphas,*params])
