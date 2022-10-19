import numpy as np
import scipy.optimize as opt
import time
import matplotlib.pyplot as plt
import sys
from dek import DEKInfinite, LinearKernel

class ExpFam(object):
    # THE NEGATIVE LOG LIKELIHOOD AND GRADIENTS OF EXPONENTIAL FAMILY
    def __init__(self, A, T, sigma, R, R_grad):
        self.A = A
        self.T = T
        self.sigma = sigma
        self.R = R
        self.R_grad = R_grad

    def nll(self, V, phi, x):
        eta = V @ phi
        d = V.shape[0]
        return np.sum(-self.T(x)*self.R(eta) + self.A(self.R(eta)))/d

    def grad(self, V, phi, x):
        eta = V @ phi
        d = V.shape[0]
        ret =  V.T @ (-self.T(x)*self.R_grad(eta) + self.sigma(eta))/d
        return ret


class GaussPrior(object):
    def __init__(self, lambd, d):
        self.lambd = lambd
        self.d = d

    def lp(self, phi):
        m = phi.shape[0]
        return self.lambd/2 * np.linalg.norm(phi)**2 * np.sqrt(m/d)

    def grad(self, phi):
        m = phi.shape[0]
        return self.lambd * phi * np.sqrt(m/d)


class Posterior(object):
    # POSTERIOR, WHOSE LOGARITHM IS LOG PRIOR + LOG LIKELIHOOD
    def __init__(self, prior, likelihood, V):
        self.prior = prior
        self.likelihood = likelihood
        self.V = V

    def nlp(self, phi, x):
        ret = self.prior.lp(phi) + self.likelihood.nll(self.V, phi, x)
        return ret

    def grad(self, phi, x):
        ret =  self.prior.grad(phi) + self.likelihood.grad(self.V, phi, x)
        return ret


class ExpectedPosterior(object):
    def __init__(self, prior, likelihood, Vlist):
        self.prior = prior
        self.likelihood = likelihood
        self.Vlist = Vlist
        self.posteriors = [Posterior(prior, likelihood, V) for V in self.Vlist]

    def nlp(self, phi, x):
        ret = 0
        for post in self.posteriors:
            ret = ret + post.nlp(phi, x)
        return ret/len(self.posteriors)

    def grad(self, phi, x):
        ret = 0
        for post in self.posteriors:
            ret += post.grad(phi, x)
        return ret/len(self.posteriors)


class SGD(object):
    def __init__(self, f):
        self.f = f

    def minimize(self, p0, step_size, x):
        for t in range(len(self.f.posteriors)):
            p0 = p0 - step_size(t+1)* self.f.posteriors[t].grad(p0, x)
        return p0


class KernelC(object):
    # AN INSTANCE OF A KERNEL AND FINITE FEATURES. LINEAR KERNEL
    def __init__(self, l, d):
        self.Q = np.random.normal(0, 1, (d, l))

    def features(self, x):
        return self.Q @ x

    def kernel(self, x, xdash):
        return (x.T @ xdash)[0,0]


class KernelThreeTypes(object):
    """
    An object with three different kernel methods:
        1. The infinite limiting kernel
        2. A randomly sampled finite m,d kernel
        3. The kernel obtained by passing the infinite limiting kernel through 
            one step of the finite kernel

        lamb (float):
            do SGD with Gaussian prior, given value of lamb, and step 
            size 1/lamb to the objective rescaled by sqrt(m).
    """
    def __init__(self, d, m, l, N, lamb, expfam_str='gauss', R_str='ident',
        step_size_factor = 1):
        self.d = d
        self.m = m
        self.l = l
        self.N = N
        self.lamb = lamb
        self.rho = lambda z: np.ones_like(z)
        self.step_size_factor = step_size_factor
        self.p0 = np.random.normal(0, 1., (self.m,))/np.sqrt(self.m)

        self._init_kernels(expfam_str, R_str)

        # Set up the kernel C and the matrix V
        self.C = KernelC(l, d)
        self.V = np.random.normal(0, 1, (d, m))
        self.Vlist = [np.random.normal(0, 1, (d, m))\
            for i in range(self.N)]

        # Setup likelihood and prior
        self.prior = GaussPrior(self.lamb, self.d)
        self._initialise_likelihood(expfam_str, R_str)
        self.lp = ExpectedPosterior(self.prior, self.likelihood, self.Vlist)

    def _init_kernels(self, expfam_str, R_str):
        self.kernel_sig = None; self.kernel_rho = None
        if R_str == 'ident':
            self.kernel_rho = lambda psi11, psi22, psi12: psi12
            if expfam_str == 'gauss':
                self.kernel_sig = lambda psi11, psi22, psi12:\
                    psi12

        if R_str == 'relu':
            cos_angle = lambda psi11, psi22, psi12: np.clip(psi12/np.sqrt(psi22*psi11),
            -1, 1)
            self.kernel_rho = lambda psi11, psi22, psi12:\
                1/(2*np.pi)*(np.pi - np.arccos(cos_angle(psi11,psi22,psi12)))
            if expfam_str == 'gauss':
                self.kernel_sig = lambda psi11, psi22, psi12:\
                    np.sqrt(psi11*psi22)/(2*np.pi)*(\
                    np.sqrt((1-cos_angle(psi11,psi22,psi12)**2))+\
                    (np.pi-np.arccos(cos_angle(psi11,psi22,psi12)))*\
                    cos_angle(psi11,psi22,psi12))

    def _initialise_likelihood(self, expfam_str, R_str):
        self.R_str = R_str
        self.expfam_str = expfam_str 

        if expfam_str == 'gauss':
            A = lambda z: z**2/2
            T = lambda z: z
            Adash = lambda z: z

        elif expfam_str == 'bernoulli':
            A = lambda z: np.logaddexp(z, 0)
            T = lambda z: z
            Adash = lambda z: 1/(1+ np.exp(-z))

        if R_str == 'ident':
            sigma = Adash
            R = lambda z: z
            R_grad =lambda z: np.ones_like(z)
        elif (R_str == 'relu') and (expfam_str == 'gauss'):
            sigma = lambda z: z*(z>0)
            R = lambda z: z*(z>0)
            R_grad = lambda z: z>0

        self.likelihood = ExpFam(A, T, sigma, R, R_grad)

    def _find_gamma(self, x):
        """ Map x to gamma """
        return self.C.features(x).reshape((-1,))

    def _find_psi(self, x, mode='sgd'):
        """ Solve m dimensional optimisation problem for psi"""
        gamma = self._find_gamma(x)
        # Define prior and likelihood
        #lp = Posterior(self.prior, self.likelihood, self.V)

        if mode == 'newton':
            psi = opt.minimize(lambda phi: self.lp.nlp(phi, gamma), 
                self.p0, 
                jac=lambda phi: self.lp.grad(phi, gamma), 
                method='BFGS')
            ret = psi.x
        elif mode == 'sgd':
            step_size = lambda t: self.step_size_factor * \
                np.sqrt(self.d)/(np.sqrt(self.m)+0.*t)/self.lamb
            ret = SGD(self.lp).minimize(self.p0, step_size, gamma)

        return ret

    def _find_limiting_kernel(self, x, xdash):
        """ Wrapper for the first type of kernel """
        c11 = self.C.kernel(x, x)
        c22 = self.C.kernel(xdash, xdash)
        c12 = self.C.kernel(x, xdash)
        
        return self._find_root(c11, c22, c12)

    def _find_root(self, c11, c22, c12):
        if self.expfam_str == 'gauss' and self.R_str == 'ident':
            lamb1 = self.lamb; lamb2 = self.lamb

            #print(lamb1*lamb2)
            limit_kernel11 = c11/(lamb1**2-1)
            limit_kernel22 = c22/(lamb2**2-1)
            limit_kernel12 = c12/(lamb1*lamb2-1)
            return limit_kernel11, limit_kernel22, limit_kernel12, lamb1, lamb2
        else:
            psi0 = self._init_psi(c11, c22, c12)
            G = lambda psi: self._g_iteration(psi[0], psi[1], psi[2], 
                c11, c22, c12)
            root = opt.root(G, psi0, method='anderson').x
            return root[0], root[1], root[2], lamb, lamb

    def _init_psi(self, c11, c22, c12):
        psi0 = [1, 1, 0]
        if self.expfam_str == 'gauss' and self.R_str == 'relu':
            psi0 = [c11/(2*self.lamb**2-1), c22/(2*self.lamb**2-1), 0]

        return psi0

    def _g_iteration(self, psi11, psi22, psi12, c11, c22, c12):
        k_rho_11 = self.k_rho(psi11, psi11, psi11)
        k_rho_22 = self.k_rho(psi22, psi22, psi22)
        k_rho_12 = self.k_rho(psi11, psi22, psi12)

        k_sig_11 = self.k_sig(psi11, psi11, psi11)
        k_sig_22 = self.k_sig(psi22, psi22, psi22)
        k_sig_12 = self.k_sig(psi11, psi22, psi12)
        
        g1 = 1/(self.lamb**2) * (c11 * k_rho_11 + k_sig_11)
        g2 = 1/(self.lamb**2) * (c22 * k_rho_22 + k_sig_22)
        g12 = 1/(self.lamb**2) * (c12 * k_rho_12 + k_sig_12)

        ret =  [g1 - psi11, g2 - psi22, g12 -psi12]
        return ret

    def k_sig(self, psi11, psi22, psi12):
        if self.kernel_sig is None:
            self.kernel_sig = MCNNKernel(self.likelihood.sigma)

        return self.kernel_sig(psi11, psi22, psi12)

    def k_rho(self, psi11, psi22, psi12):
        if self.kernel_rho is None:
            self.kernel_rho = MCNNKernel(self.rho)

        return self.kernel_rho(psi11, psi22, psi12)

    def find_limiting_kernel(self, x, xdash):
        """ The first type of kernel """
        t0 = time.time()
        ret =  self._find_limiting_kernel(x, xdash)[2]

        cost = time.time() - t0
        return ret, cost

    def find_finite_kernel(self, x, xdash):
        """ The second type of kernel"""
        t0 = time.time()

        psi = self._find_psi(x, mode='sgd')
        psidash = self._find_psi(xdash, mode='sgd')
        ret = psi.T @ psidash
        
        cost = time.time() - t0
        return ret, cost
    
    def _find_r_rdash(self, x, xdash):
        limit_kernel11, limit_kernel22, limit_kernel12, lamb1, lamb2 = \
        self._find_limiting_kernel(x, xdash)

        rho = np.clip(limit_kernel12/np.sqrt(limit_kernel11*limit_kernel22), -1, 1)

        r = np.zeros((self.m,)); rdash = np.zeros((self.m,))
        r[0] = np.sqrt(limit_kernel11)
        rdash[0] = rho; rdash[1] = np.sqrt(1 - rho**2)
        rdash = rdash * np.sqrt(limit_kernel22)
        return r, rdash, lamb1, lamb2

    def find_one_step_kernel(self, x, xdash):
        """ The third type of kernel""" 
        t0 = time.time()
        r, rdash, lamb1, lamb2 = self._find_r_rdash(x, xdash)

        gamma = self._find_gamma(x)
        gammadash = self._find_gamma(xdash)

        kd_feat_r = self.likelihood.grad(self.V, r, gamma)
        kd_feat_rdash = self.likelihood.grad(self.V, rdash, gammadash)

        kd_res = kd_feat_r.T@kd_feat_rdash/(lamb1*lamb2)*self.d/self.m
        cost = time.time() - t0

        return kd_res, cost

class MCNNKernel(object):
    def __init__(self, activation, num_samples=100):
        self.activation = activation
        self.num_samples = num_samples

    def __call__(self, psi11, psi22, psi12):
        cov = np.asarray([[psi11, psi12], [psi12, psi22]])
        # Careful to handle numerically non-PSD matrices
        chi = np.random.multivariate_normal([0,0], 
            cov,
            self.num_samples) # (num_samples, 2)
        acts = self.activation(chi)
        return np.average(acts[:,0]*acts[:,1])

def plot_kernel_matrix(K, fname):
    plt.imshow(K)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    # Simulation Parameters
    d       = int(sys.argv[1]) #300 runs in okay time
    m       = int(d**(3/2)) # choose m >> d
    l       = 2
    N       = int(sys.argv[2]) # 100
    lamb    = 6
    expfam_type = 'gauss'
    R_str   = 'relu'
    output_dir = 'outputs/change_d_relu_new/'
    step_size_factor = float(sys.argv[3])#0.00001

    print(d)
    print(N)
    print(step_size_factor)

    x1 = np.linspace(-5, 5, 10)
    x2 = np.linspace(-5, 5, 10)
    xarr = np.transpose([np.tile(x1, len(x2)), np.repeat(x1, len(x2))])

    K = KernelThreeTypes(d, m, l, N, lamb, expfam_type, R_str, step_size_factor=step_size_factor)
    Kpsi = DEKInfinite('relu', 'heavy', LinearKernel(), lamb, 10)

    K_infinite = np.zeros((len(xarr), len(xarr)))
    K_finite = np.zeros((len(xarr), len(xarr)))
    K_one_step = np.zeros((len(xarr), len(xarr)))

    T_infinite = np.zeros((len(xarr), len(xarr)))
    T_finite = np.zeros((len(xarr), len(xarr)))
    T_one_step = np.zeros((len(xarr), len(xarr)))

    for i, x in enumerate(xarr):
        for j in range(i, len(xarr)):
            xdash = xarr[j,:]
            x = x.reshape((-1,1))
            xdash = xdash.reshape((-1,1))

            #k_psi, t_psi        = K.find_limiting_kernel(x, xdash)
            k_finite, t_finite  = K.find_finite_kernel(x, xdash)
            #k_one_step, t_one   = K.find_one_step_kernel(x, xdash)
            print(k_finite)
            #print(k_one_step)
            print(i)
            print(j)
            print('')

            #K_infinite[i,j] = k_psi
            K_finite[i,j] = k_finite
            #K_one_step[i,j] = k_one_step

            #T_infinite[i,j] = t_psi
            T_finite[i,j] = t_finite
            #T_one_step[i,j] = t_one

            #K_infinite[j,i] = k_psi
            K_finite[j,i] = k_finite
            #K_one_step[j,i] = k_one_step

            #T_infinite[j,i] = t_psi
            T_finite[j,i] = t_finite
            #T_one_step[j,i] = t_one
    
    prefix = ''

    K_infinite = Kpsi(xarr, xarr)


    np.save(output_dir + prefix + 'kpsi_' + str(d) + '_' + str(N) + '_' + \
        str(step_size_factor) + '.npy', K_infinite)
    np.save(output_dir + prefix + 'kfinite_' + str(d) + '_' + str(N) + '_' + \
        str(step_size_factor) + '.npy', K_finite)
    #np.save(output_dir + prefix + 'kone_' + str(d) + '_' + str(N) + '_' + \
    #    str(step_size_factor) + '.npy', K_one_step)

    np.save(output_dir + prefix + 'tpsi_' + str(d) + '_' + str(N) + '_' + \
        str(step_size_factor) + '.npy', T_infinite)
    np.save(output_dir + prefix + 'tfinite_' + str(d) + '_' + str(N) + '_' + \
        str(step_size_factor) + '.npy', T_finite)
    #np.save(output_dir + prefix + 'tone_' + str(d) + '_' + str(N) + '_' + \
    #    str(step_size_factor) + '.npy', T_one_step)

    plot_kernel_matrix(K_infinite, '/home/tsu007/dek/' +output_dir + prefix + 'kpsi_' + str(d) + '_' + str(N) + \
        '_' + str(step_size_factor) + '.png')
    plot_kernel_matrix(K_finite, '/home/tsu007/dek/'+output_dir + prefix + 'kfinite_' + str(d) + '_' + str(N) + \
        '_' + str(step_size_factor) + '.png')
    #plot_kernel_matrix(K_one_step, '/home/tsu007/dek/'+output_dir + prefix + 'kone_' + str(d) + '_' + str(N) + \
    #    '_' + str(step_size_factor) + '.png')

