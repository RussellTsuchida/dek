from kernels import DEK, DeepNNKInfinite, NTKInfinite, RBFKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import sklearn.datasets as datasets
import sklearn.preprocessing as preprocessing
import numpy as np
import sys


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
        c_s (float): lengthscale parameter for c kernel
        c_scale (float): scaling parameter for c kernel
    """
    def __init__(self, kernel_str, T, sigma_str, rho_str, lamb,
        c_kernel, scale, preprocess, reg_strength, c_ls, c_scale):
        self._init_kernel(kernel_str, T, sigma_str, rho_str, lamb, c_kernel, c_ls, c_scale)
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
        c_ls, c_scale):
        self.kernel_str = kernel_str
        self.T          = T
        self.sigma_str  = sigma_str
        self.rho_str    = rho_str
        self.lamb       = lamb
        self.c_kernel   = c_kernel
        self.c_kernel_map= c_kernel(c_ls, c_scale)
        self.c_ls       = c_ls
        self.c_scale    = c_scale

        if kernel_str == 'nnk':
            self.kernel = DeepNNKInfinite(sigma_str, L=T)
        elif kernel_str == 'ntk':
            self.kernel = NTKInfinite(sigma_str, rho_str, L=T)
        elif kernel_str == 'dek':
            self.kernel = DEK(sigma_str, rho_str, self.c_kernel_map, lamb, T)

    def fit(self, X, y, sample_weight=None):
        if not (self.preprocess == 'none'):
            self.pretransform.fit(X)
            X = self.pretransform.transform(X)
        X = X * self.scale
        self.X_train = X
        Kxx = self.kernel(X,X)
        self.algorithm.fit(Kxx, y, sample_weight)
        return self

    def get_params(self, deep=False):
        dict_ = {'kernel_str': self.kernel_str, 'T': self.T, 
            'sigma_str': self.sigma_str, 'rho_str': self.rho_str, 
            'lamb':self.lamb, 'c_kernel':self.c_kernel, 'scale':self.scale,
            'preprocess':self.preprocess, 'reg_strength':self.reg_strength,
            'c_ls':self.c_ls, 'c_scale':self.c_scale}
        return dict_

    def predict(self, X):
        if not (self.preprocess == 'none'):
            self.pretransform.fit(X)
            X = self.pretransform.transform(X) 
        X = X*self.scale
        K_xstar = self.kernel(X, self.X_train)
        return self.algorithm.predict(K_xstar)

    def score(self, X,y,sample_weight=None):
        if not (self.preprocess == 'none'):
            self.pretransform.fit(X)
            X = self.pretransform.transform(X) 
        X = X*self.scale
        K_xstar = self.kernel(X, self.X_train)
        return self.algorithm.score(K_xstar,y, sample_weight)

    def set_params(self, **params):
        self._init_kernel(params['kernel_str'], params['T'], 
            params['sigma_str'], params['rho_str'], params['lamb'], 
            params['c_kernel'], params['c_ls'], params['c_scale'])
        self._init_algorithm(params['reg_strength'])
        self._init_preprocessor(params['scale'], params['preprocess'])
        return self

if __name__ == '__main__':
    KERNEL_TYPE     = sys.argv[1] #'dek' or 'ntk' or 'nnk' or 'rbf'
    DATASET         = sys.argv[2]
    SHUFFLE_SEED    = int(sys.argv[3])
    TRAIN_SPLIT     = 0.8

    

    # Load the dataset into training and testing sets
    if DATASET == 'diabetes':
        X, y = datasets.load_diabetes(return_X_y = True)
    elif DATASET == 'boston':
        X, y = datasets.load_boston(return_X_y = True)

    y = (y - np.mean(y))/np.std(y)
    X, y = shuffle(X, y, random_state=SHUFFLE_SEED)

    idx = int(TRAIN_SPLIT * X.shape[0])
    X_test = X[idx:,:]; y_test = y[idx:]
    X = X[:idx,:]; y = y[:idx]


    # Hyperparameter grid to do CV grid search over
    lengthscale_list    = [0.1, 0.5, 1, 2]
    reg_list            = [0.05, 0.1, 0.5]
    lamb_list           = [1, 2, 4, 8]
    scale_list          = [0.1, 0.5, 1, 2, 4]
    c_scale_list        = [0.5, 1, 2, 4]
    T_list              = [2, 3, 4, 5]
    if KERNEL_TYPE == 'dek':
        T_list = [None] + T_list
    
    # Set up the model
    if KERNEL_TYPE == 'rbf':
        # Convert units of hyperparameters
        gamma_list = [1/(2*ls**2) for ls in lengthscale_list]
        alpha_list = [lamb / 2 for lamb in reg_list]

        gs_krr = GridSearchCV(
            KernelRidge(kernel="rbf", gamma=0.1),
            param_grid={"alpha": alpha_list, "gamma": gamma_list},
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
            'c_kernel':[RBFKernel],
            'preprocess':['none'],
            'c_ls':lengthscale_list,
            'c_scale':c_scale_list}
        krr = KrrWithScaling(KERNEL_TYPE, 2, 'relu', 'heavy', 2, RBFKernel, 10,
        'white', 0.1, 1, 1)
        gs_krr = GridSearchCV(krr, param_grid=param_grid, verbose=0, n_jobs=-1)
    
    
    
    ## Run the grid search
    gs_krr.fit(X,y)
    print(gs_krr.best_score_)
    print(gs_krr.best_params_)


    # Evaluate the model on test data
    RMSE = np.sqrt(np.average( (y_test - gs_krr.predict(X_test))**2))
    print(RMSE)



