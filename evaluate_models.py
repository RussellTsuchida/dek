from kernels import DEK, DeepNNKInfinite, NTKInfinite, RBFKernel, LinearKernel, ReLUMean, ShallowNNKInfinite, RBFMean
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import sklearn.datasets as datasets
import sklearn.preprocessing as preprocessing
import numpy as np
import sys
import pandas as pd
import os
import time

class KrrWithScaling(BaseEstimator):
    """
    Wrapper for Scikit Learn's model (e.g. kernel ridge regression model).

    Args:
        kernel_str (str): 'nnk', 'ntk' or 'dek'
        T (int or None): For nnk and ntk, the number of layers. For dek, the 
            number of sgd iterations. For dek only, if None, use anderson
            fixed point solver.
        sigma_str (str or func): String of a prebuilt sigma. 
            Use 'relu', 'heavy', 'linear' or 'constant1'. If a func, use 
            empirical kernel using this function as the activation.
        rho_str (str or func): String of a prebuilt rho. 
            Use 'relu', 'heavy', 'linear' or 'constant1'. If a func, use 
            empirical kernel using this function as the activation. For dek,
            this represents rho and for ntk this represents sigma dot.
            Unused for nnk.
        lambd (float): For dek kernel. Ignored for nnk and ntk.
        c_kernel (func): The explicit kernel C. Unused for nnk and ntk.
        scale (float): Scale the input data X by this factor
        preprocess (str): 'standard' or 'white'. Preprocess the input data to
            have zero-mean variance (standard), or additionall also have
            identity empirical covariance matrix.
        reg_strength (float): Regularisation strength for kernel ridge 
            regression.
        c_scale (float): scaling parameter for c kernel
    """
    def __init__(self, kernel_str, T, sigma_str, rho_str, lamb,
        c_kernel, scale, preprocess, reg_strength, c_scale, mean_fun=None, 
        bias=0):
        self._init_kernel(kernel_str, T, sigma_str, rho_str, lamb, c_kernel, 
            c_scale, mean_fun, bias)
        self._init_algorithm(reg_strength)
        self._init_preprocessor(scale, preprocess)

    def _init_preprocessor(self, scale, preprocess):
        self.scale = scale
        self.preprocess = preprocess
        if preprocess == 'white':
            self.pretransform = PCA(whiten=True)
        elif preprocess == 'standard':
            self.pretransform = preprocessing.StandardScaler()
        elif preprocess == 'none':
            self.pretransform = lambda x: x
            

    def _init_algorithm(self, reg_strength):
        self.reg_strength = reg_strength
        self.algorithm = KernelRidge(alpha=reg_strength, kernel='precomputed')

    def _init_kernel(self, kernel_str, T, sigma_str, rho_str, lamb, c_kernel,
        c_scale, mean_fun, bias):
        self.kernel_str = kernel_str
        self.T          = T
        self.sigma_str  = sigma_str
        self.rho_str    = rho_str
        self.lamb       = lamb
        self.c_kernel   = c_kernel
        self.bias       = bias
        self.mean_fun = mean_fun
        if c_kernel == 'zero':
            self.c_kernel_map = 'zero'
        elif c_kernel == 'relu':
            self.c_kernel_map = ShallowNNKInfinite('relu').kernel
        self.c_scale    = c_scale

        if kernel_str == 'nnk':
            self.kernel = DeepNNKInfinite(sigma_str, L=T)
        elif kernel_str == 'ntk':
            self.kernel = NTKInfinite(sigma_str, rho_str, L=T)
        elif kernel_str == 'dek':
            self.kernel = DEK(sigma_str, rho_str, self.c_kernel_map, lamb, T, 
                mean_fun=mean_fun, c_scale=c_scale)

    def fit(self, X, y, sample_weight=None):
        if not (self.preprocess == 'none'):
            self.pretransform.fit(X)
            X = self.pretransform.transform(X)
        X = X * self.scale
        self.X_train = self._bias_input_X(X)
        Kxx = self.kernel(self.X_train,self.X_train)
        self.algorithm.fit(Kxx, y, sample_weight)
        return self

    def get_params(self, deep=False):
        dict_ = {'kernel_str': self.kernel_str, 'T': self.T, 
            'sigma_str': self.sigma_str, 'rho_str': self.rho_str, 
            'lamb':self.lamb, 'c_kernel':self.c_kernel, 'scale':self.scale,
            'preprocess':self.preprocess, 'reg_strength':self.reg_strength,
            'c_scale':self.c_scale, 'mean_fun':self.mean_fun, 'bias':self.bias}
        return dict_

    def predict(self, X):
        if not (self.preprocess == 'none'):
            self.pretransform.fit(X)
            X = self.pretransform.transform(X) 
        X = self._bias_input_X(X*self.scale)
        K_xstar = self.kernel(X, self.X_train)
        return self.algorithm.predict(K_xstar)

    def score(self, X,y,sample_weight=None):
        if not (self.preprocess == 'none'):
            self.pretransform.fit(X)
            X = self.pretransform.transform(X) 
        X = self._bias_input_X(X*self.scale)
        K_xstar = self.kernel(X, self.X_train)
        return self.algorithm.score(K_xstar,y, sample_weight)

    def set_params(self, **params):
        self._init_kernel(params['kernel_str'], params['T'], 
            params['sigma_str'], params['rho_str'], params['lamb'], 
            params['c_kernel'], params['c_scale'],
            params['mean_fun'], params['bias'])
        self._init_algorithm(params['reg_strength'])
        self._init_preprocessor(params['scale'], params['preprocess'])
        return self

    def _bias_input_X(self, X):
        ret = np.hstack((X, np.ones((X.shape[0], 1))*self.bias))
        return ret

class ExperimentData(object):
    def __init__(self, fname):
        self._init_dataframe(fname)

    def _init_dataframe(self, fname):
        self.fname = fname
        try:
            self.pd_array = pd.read_csv(fname)
        except:
            self.pd_array = pd.DataFrame(columns=\
                ['kernel','dataset','seed','rmse','c_kernel'])

    def save_result(self, kernel_type, dataset, seed, rmse, c_kernel):
        data = {'kernel':kernel_type, 'dataset':dataset,
            'seed':seed,'rmse':rmse,'c_kernel':c_kernel}
        self.pd_array = self.pd_array.append(pd.DataFrame(\
            index=(self.pd_array.index.max()+1,),
            columns=self.pd_array.columns,
            data=data))
        self.pd_array.drop_duplicates(inplace=True)
        self.pd_array.to_csv(self.fname, index=False)


if __name__ == '__main__':
    KERNEL_TYPE     = sys.argv[1] #'dek' or 'ntk' or 'nnk' or 'rbf'
    DATASET         = sys.argv[2]
    SHUFFLE_SEED    = int(sys.argv[3])
    TRAIN_SPLIT     = 0.8
    OUTPUT_CSV      = '/home/tsu007/dek/outputs/benchmark/results.csv'
    DATA_DIR        = '/home/tsu007/dek/data/'

    # Load the dataset into training and testing sets
    if DATASET == 'diabetes':
        X, y = datasets.load_diabetes(return_X_y = True)
        X = X * np.sqrt(X.shape[0])
    elif DATASET == 'boston':
        X, y = datasets.load_boston(return_X_y = True)
    elif DATASET == 'california':
        X, y = datasets.fetch_california_housing(return_X_y=True)
    elif DATASET == 'concrete':
        dat = np.genfromtxt(DATA_DIR + 'concrete.csv', delimiter=',',dtype=np.float32)
        X = dat[:,:8]; y = dat[:,8]
    elif DATASET == 'energy1':
        dat = np.genfromtxt(DATA_DIR + 'energy.csv', delimiter=',',dtype=np.float32)
        X = dat[:,:8]; y = dat[:,8]
    elif DATASET == 'energy2':
        dat = np.genfromtxt(DATA_DIR + 'energy.csv', delimiter=',',dtype=np.float32)
        X = dat[:,:8]; y = dat[:,9]
    elif DATASET == 'wine':
        dat = np.genfromtxt(DATA_DIR + 'wine.csv', delimiter=',',dtype=np.float32)
        X = dat[:,:11]; y = dat[:,11]
    elif DATASET == 'yacht':
        dat = np.genfromtxt(DATA_DIR + 'yacht.csv', delimiter=',',dtype=np.float32)
        X = dat[:,:6]; y = dat[:,6]
    elif DATASET == 'mpg':
        dat = np.genfromtxt(DATA_DIR + 'mpg.csv', delimiter=',',dtype=np.float32,
            usecols=[0,1,2,3,4,5,6,7])
        X = dat[:,1:8]; y = dat[:,0]

    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    X, y = shuffle(X, y, random_state=SHUFFLE_SEED)
    #if DATASET == 'california':
    #    X = X[:2000,:]; y = y[:2000]

    idx = int(TRAIN_SPLIT * X.shape[0])
    X_test = X[idx:,:]; y_test = y[idx:]
    X = X[:idx,:]; y = y[:idx]
    centre_y = np.mean(y)
    scale_y = np.std(y)
    y = (y - centre_y)/scale_y
    ###########################################################################

    # Hyperparameter grid to do CV grid search over
    reg_list            = [0.05, 0.1, 0.5]
    scale_list          = [0.5, 1, 2, 4, 8]
    T_list              = [2, 3, 4, 5]
    bias_list           = [-1, -0.1, 0, 0.1, 1]

    if KERNEL_TYPE == 'dek':
        if X.shape[0] < 500:
            T_list = [None] + T_list
        c_scale_list        = [0.1, 0.5, 1, 2] # 'Dont use zero. Instead, set c_kernel to 'zero'
        lamb_list           = [1, 2, 4]
        c_kernel_list = ['relu']

    if KERNEL_TYPE == 'dek_zero':
        c_scale_list = [1]
        lamb_list           = [1]
        c_kernel_list = ['zero']
        KERNEL_TYPE = 'dek'

    if (KERNEL_TYPE == 'ntk') or (KERNEL_TYPE == 'nnk'):
        c_scale_list        = [1]
        lamb_list           = [1]
        c_kernel_list = ['none']

    # Set up the model
    if KERNEL_TYPE == 'rbf':
        lengthscale_list    = [0.5, 1, 2, 4, 8, 16]
        # Convert units of hyperparameters
        gamma_list = [1/(2*ls**2) for ls in lengthscale_list]

        gs_krr = GridSearchCV(
            KernelRidge(kernel='rbf', gamma=0.1),
            param_grid={"alpha": reg_list, "gamma": gamma_list},
            n_jobs=-1)
    else:
        param_grid={\
            'lamb': lamb_list, 
            'scale': scale_list,
            'reg_strength':reg_list, 
            'kernel_str': [KERNEL_TYPE],
            'T':T_list,
            'sigma_str':['relu'],
            'rho_str':['heavy'],
            'c_kernel':c_kernel_list,#[RBFKernel],#['relu'],#'zero'
            'preprocess':['none'],
            'c_scale':c_scale_list,
            'mean_fun':[ReLUMean()],
            'bias':bias_list}#[ReLUMean()] [None]}
        krr = KrrWithScaling(KERNEL_TYPE, 2, 'relu', 'heavy', 2, c_kernel_list[0], 10,
        'white', 0.1, 1, mean_fun=None, bias=0)

        n_jobs = -1
        if X.shape[0] > 3000:
            n_jobs = 2
        gs_krr = GridSearchCV(krr, param_grid=param_grid, verbose=0, n_jobs=n_jobs)
    ###########################################################################

    ## Run the grid search
    gs_krr.fit(X,y)

    # Evaluate the model on test data
    y_pred = gs_krr.predict(X_test)*scale_y + centre_y
    RMSE = np.sqrt(np.average( (y_test - y_pred)**2))

    print(gs_krr.best_score_)
    print(gs_krr.best_params_)
    print(RMSE)
    c_kernel = 'none'
    if KERNEL_TYPE == 'dek':
        c_kernel = gs_krr.best_params_['c_kernel']

    import portalocker
    with portalocker.Lock(OUTPUT_CSV, 'rb+') as fh:
        # Load the results table
        results = ExperimentData(OUTPUT_CSV)
        results.save_result(KERNEL_TYPE, DATASET, SHUFFLE_SEED, RMSE, c_kernel)
        fh.flush()
        os.fsync(fh.fileno())

