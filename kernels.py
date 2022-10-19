import numpy as np
from abc import abstractmethod, ABC
import scipy.spatial as sp
import scipy.optimize as opt
from sklearn.kernel_ridge import KernelRidge
import sklearn.datasets as datasets
import sklearn.preprocessing as preprocess

PREBUILT = ['relu', 'heavy', 'linear', 'constant1']

class Kernel(ABC):
    @abstractmethod
    def __call__(self, X, Y=None):
        """
        Args:
            X (nparray(float)): (n1, d) array representing n1 samples of d
                dimensional inputs.
            Y (nparray(float)): (n2, d) array representing n2 samples of d
                dimensional inputs.
        """
        pass


class InputKernel(Kernel):
    def __init__(self):
        pass
    def __call__(self, X, Y):
        Xnorm2 = (np.linalg.norm(X, axis=1)**2).reshape((-1,1))
        if Y is None:
            Ynorm2 = Xnorm2
            inner = X @ X.T 
        else:
            Ynorm2 = (np.linalg.norm(Y, axis=1)**2).reshape((-1,1))
            inner = X @ Y.T
        return Xnorm2, Ynorm2, inner


class ShallowNNKInfinite(Kernel):
    def __init__(self, sigma_str):
        assert sigma_str in PREBUILT
        self.sigma_str = sigma_str
        self.input_ker = InputKernel()

    def __call__(self, X, Y):
        k11, k22, k12 = self.input_ker(X,Y)

        return self.kernel(k11, k22, k12)[2]
    
    def kernel(self, k11, k22, k12):
        if self.sigma_str == 'relu':
            norm_prod = np.sqrt(k11 @ k22.T)
            div = np.divide(k12, norm_prod, out=np.zeros_like(k12), where=norm_prod!=0)
            cos = np.clip(div, -1, 1)
            sin = np.sqrt(np.clip(1-cos**2, 0, 1))
            theta = np.arccos(cos)
            k12 = norm_prod/(2*np.pi)*(sin + (np.pi-theta)*cos)
        elif self.sigma_str == 'heavy':
            norm_prod = np.sqrt(k11 @ k22.T)
            div = np.divide(k12, norm_prod, out=np.zeros_like(k12), where=norm_prod!=0)
            cos = np.clip(div, -1, 1)
            k12 = 1/(2*np.pi)*(np.pi - np.arccos(cos))
        elif self.sigma_str == 'linear':
            pass
        elif self.sigma_str == 'constant1':
            k12 = np.ones_like(k12)
        return self.kernel11(k11), self.kernel11(k22), k12

    def kernel11(self, k11):
        if self.sigma_str == 'relu':
            return k11/2
        elif self.sigma_str == 'heavy':
            return 1/2*np.ones_like(k11)
        elif self.sigma_str == 'linear':
            return k11
        elif self.sigma_str == 'constant1':
            return np.ones_like(k11)


class ShallowNNKFinite(Kernel):
    def __init__(self, sigma_str, width=200, d=2):
        self._init_empirical_flag(sigma_str, width, d)

    def _init_empirical_flag(self, sigma, width, d):
        self.sigma = sigma
        self.W = np.random.normal(0, 1, (width, d))

    def kernel(self, X, Y=None):
        preactX = self.W @ X.T # (width, n1)
        if Y is None:
            preactY = preactX
        else:
            preactY = self.W @ Y.T # (width, n2)

        s = self.W.shape[0]

        hx = self.sigma(preactX)/np.sqrt(s)
        hy = self.sigma(preactY)/np.sqrt(s)

        inner = hx.T @ hy #(n1, n2)
        return hx.T, hy.T, inner

    def kernel11(self, X):
        k11 = np.linalg.norm(X, axis=1).reshape((-1,1))
        if self.sigma_str == 'relu':
            return k11/2
        elif self.sigma_str == 'heavy':
            return 1/2*np.ones_like(k11)
        elif self.sigma_str == 'linear':
            return k11
        elif self.sigma_str == 'constant1':
            return np.ones_like(k11)

    def __call__(self, X, Y):
        return self.kernel(X, Y)[2]
        

class DeepNNKInfinite(Kernel):
    def __init__(self, sigma_str, L):
        self.input_kernel = InputKernel()
        self.k_list = []
        for l in range(L):
            k = ShallowNNKInfinite(sigma_str)
            self.k_list.append(k)

    def __call__(self, X, Y):
        k11, k22, k12 = self.input_kernel(X, Y)
        for k in self.k_list:
            k11, k22, k12 = k.kernel(k11, k22, k12)
        return k12


class DeepNNKFinite(Kernel):
    def __init__(self, sigma_str, L, width=200, d=2):
        self.k_list = []
        num_inputs = d
        for l in range(L):
            k = ShallowNNKFinite(sigma_str, width, num_inputs)
            self.k_list.append(k)
            num_inputs = width

    def __call__(self, X, Y):
        for k in self.k_list:
            X, Y, k12 = k.kernel(X, Y)
        return k12


class NTKInfinite(Kernel):
    def __init__(self, sigma_str, sigma_dot_str, L):
        self.input_kernel = InputKernel()
        self.k_list = []
        self.kdot_list = []
        for l in range(L):
            k = ShallowNNKInfinite(sigma_str)
            kdot = ShallowNNKInfinite(sigma_dot_str)
            self.k_list.append(k)
            self.kdot_list.append(kdot)

    def __call__(self, X, Y):
        k11, k22, k12 = self.input_kernel(X, Y)
        ntk = k12.copy()
        for k, kdot in zip(self.k_list, self.kdot_list):
            _, _, kdot12 = kdot.kernel(k11, k22, k12)
            k11, k22, k12 = k.kernel(k11, k22, k12)
            ntk = ntk*kdot12 + k12
        return ntk


class NTKFinite(Kernel):
    def __init__(self, sigma, sigma_dot, L, width=200, d=2):
        self.k_list = []
        self.kdot_list = []
        num_inputs = d
        for l in range(L):
            k = ShallowNNKFinite(sigma, width, num_inputs)
            kdot = ShallowNNKFinite(sigma_dot, width, num_inputs)
            self.k_list.append(k)
            self.kdot_list.append(kdot)
            num_inputs = width

    def __call__(self, X, Y):
        ntk = X @ Y.T
        for k, kdot in zip(self.k_list, self.kdot_list):
            _, _, kdot12 = kdot.kernel(X, Y)
            X, Y, k12 = k.kernel(X, Y)
            ntk = ntk*kdot12 + k12
        return ntk
            

class DEK(Kernel):
    def __init__(self, sigma_str, rho_str, C, lambd, T=None):
        self.sigma_str = sigma_str
        self.rho_str = rho_str
        if sigma_str in PREBUILT:
            self.k_sigma = ShallowNNKInfinite(sigma_str)
        else:
            self.k_sigma = ShallowNNKFinite(sigma_str, width=200, d=2)
        if rho_str in PREBUILT:
            self.k_rho = ShallowNNKInfinite(rho_str)
        else:
            self.k_rho = ShallowNNKFinite(rho_str, width=200, d=2)

        self.C = C
        self.lamb = lambd
        self.T = T

        self.input_ker = InputKernel()

    def g11(self, psi11, C11, psi11_shape):
        """
        Accept a vector and return a vector
        """
        psi11 = psi11.reshape(psi11_shape)
        if not ((self.rho_str in PREBUILT) and (self.sigma_str in PREBUILT)):
            X = np.hstack((psi11, np.zeros_like(psi11)))

        if self.rho_str in PREBUILT:
            rho_eval = self.k_rho.kernel11(psi11)
        else:
            rho_eval = self.k_rho.kernel11(X)
        if self.sigma_str in PREBUILT:
            sigma_eval = self.k_sigma.kernel11(psi11)
        else:
            sigma_eval = self.k_sigma.kernel11(X)

        ret = 1/self.lamb**2 * (C11*rho_eval + sigma_eval) - psi11

        return ret.flatten()

    def g12(self, psi11, psi22, psi12, C12, psi12_shape):
        psi12 = psi12.reshape(psi12_shape)

        if not ((self.rho_str in PREBUILT) and (self.sigma_str in PREBUILT)):
            #TODO
            #norm_prod = np.sqrt(psi11 @ psi22.T)
            #div = np.divide(psi12, norm_prod, out=np.zeros_like(psi12), where=norm_prod!=0)
            #cos = np.clip(div, -1, 1)
            #X = np.hstack((psi11, np.zeros_like(psi11)))
            pass

        ret = 1/self.lamb**2 * (C12*\
            self.k_rho.kernel(psi11, psi22, psi12)[2] + \
            self.k_sigma.kernel(psi11, psi22, psi12)[2]) - psi12
        return ret.flatten()

    def __call__(self, X, Y):
        k11, k22, k12 = self.input_ker(X,Y)
        C11, C22, C12 = self.C(k11, k22, k12)

        psi11_0, psi22_0, psi12_0 = self._init_guess(C11, C22, C12)
        
        if self.T is None:
            # Note: We may solve for the squared norms first, then use that 
            # solution to solve the inner product
            psi11 = opt.root(lambda p11: self.g11(p11, C11, psi11_0.shape), 
                psi11_0.flatten(), method='anderson')
            psi11 = psi11.x.reshape((psi11_0.shape))
            psi22 = opt.root(lambda p22: self.g11(p22, C22, psi22_0.shape), 
                psi22_0.flatten(), method='anderson')
            psi22 = psi22.x.reshape((psi22_0.shape))
            psi12 = opt.root(lambda p12: self.g12(psi11, psi22, p12, C12, 
                psi12_0.shape), psi12_0.flatten(), method='anderson')
            psi12 = psi12.x.reshape((psi12_0.shape))
        else:
            psi12 = self._naive_root_finder(psi11_0,
                psi22_0,
                psi12_0,
                C11, C22, C12)

        return psi12

    def _naive_root_finder(self, psi11, psi22, psi12, C11, C22, C12):
        shape11 = psi11.shape; shape22 = psi22.shape; shape12 = psi12.shape
        psi11 = psi11.flatten(); psi22 = psi22.flatten(); psi12 = psi12.flatten()
        for t in range(self.T):
            psi11_ = psi11.reshape(shape11); psi22_ = psi22.reshape(shape22)
            psi12 = self.g12(psi11_, psi22_, psi12, C12, shape12) + psi12
            psi11 = self.g11(psi11, C11, shape11) + psi11
            psi22 = self.g11(psi22, C22, shape22) + psi22
        return psi12.reshape(shape12)

    def _init_guess(self, C11, C22, C12):
        ret = [np.ones_like(C11), np.ones_like(C22), np.zeros_like(C12)]

        if (self.sigma_str == 'relu') and (self.rho_str == 'heavy'):
            ret[0] = C11/(2*self.lamb**2-1)
            ret[1] = C22/(2*self.lamb**2-1)
        
        return ret


class LinearKernel(object):
    def __init__(self):
        pass

    def __call__(self, k11, k22, k12):
        return k11, k22, k12


class RBFKernel(object):
    def __init__(self, c_ls=1, c_scale=1):
        self.c_ls = c_ls
        self.c_scale = c_scale

    def __call__(self, k11, k22, k12):
        k11_ = np.tile(k11, (1, k22.shape[0]))
        k22_ = np.tile(k22, (1, k11.shape[0])).T
        return self.c_scale*np.ones_like(k11), self.c_scale*np.ones_like(k22),\
            self.c_scale*np.exp(-(k11_ + k22_ - 2*k12)/(2*self.c_ls**2))



if __name__=='__main__':
    N = 3
    d = 2
    K1 = ShallowNNKFinite(lambda z: z*(z>0), width=1000, d=d)
    K2 = ShallowNNKInfinite('relu')
    K3 = DeepNNKFinite(lambda z: z*(z>0), L=2, width=1000, d=d)
    K4 = DeepNNKInfinite('relu', L=2)

    K5 = NTKInfinite('relu', 'heavy', L=2)
    K6 = NTKFinite(lambda z: z*(z>0), lambda z: z>0, L=2, width=1000, d=d)

    K7 = DEK('relu', 'heavy', RBFKernel, 2,10)
    K8 = DEK('relu', 'heavy', RBFKernel, 2)
    x = np.random.normal(0, 5, (N, d))
    x_star1 = np.linspace(-5, 5, 100)
    x_star = np.transpose([np.tile(x_star1, len(x_star1)), np.repeat(x_star1, len(x_star1))])
    y = x[:,0]
    print(K1(x,x))
    print(K2(x,x))
    print(K3(x, x))
    print(K4(x, x))
    print(K5(x, x))
    print(K6(x, x))
    print(K7(x, x))
    print(K8(x, x))
    print('')
    
    """


    x, y = datasets.load_diabetes(return_X_y = True)
    x = 10*x
    print(K7(x, x))
    print(K8(x, x))

    scaler = preprocess.StandardScaler()
    scaler.fit(y.reshape((-1,1)))
    y = scaler.transform(y.reshape((-1,1))).reshape((-1,))

    # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-ridge-regression-py 
    krr1 = KernelRidge(0.1, kernel='precomputed')
    krr2 = KernelRidge(0.1, kernel='rbf')
    
    K_xx = K7(x, x)
    krr1.fit(K_xx, y)
    print(krr1.score(K_xx, y))

    krr2.fit(x, y)
    print(krr2.score(x, y))

    print(K_xx)
    """
