import numpy as np
import scipy.optimize as opt

class ExpFam(object):
    # THE NEGATIVE LOG LIKELIHOOD AND GRADIENTS OF EXPONENTIAL FAMILY
    def __init__(self, A, T, A_grad, A_hess):
        self.A = A
        self.T = T
        self.A_grad = A_grad
        self.A_hess = A_hess

    def nll(self, V, phi, x):
        eta = V @ phi
        return np.sum(-self.T(x)*eta + self.A(eta))

    def grad(self, V, phi, x):
        eta = V @ phi
        ret =  V.T @ (-self.T(x) + self.A_grad(eta))
        return ret

    def hess(self, V, phi, x):
        eta = V @ phi
        return V.T @ np.diag(self.A_hess(eta)) @ V

    # A fake gradient. the one used to define k tilde.
    def _grad(self, V, phi, x):
        eta = V @ phi
        ret =  -self.T(x) + self.A_grad(eta)
        return ret


class GaussPrior(object):
    # THE NEGATIVE LOG PRIOR AND GRADIENTS OF UNIFORM PRIOR ON HYPERSPHERE
    def __init__(self, lambd, d):
        self.lambd = lambd
        self.d = d

    def lp(self, phi):
        m = phi.shape[0]
        return self.lambd/2 * np.linalg.norm(phi)**2 * np.sqrt(self.d/m)

    def grad(self, phi):
        m = phi.shape[0]
        return self.lambd * phi * np.sqrt(self.d/m)

    def hess(self, phi):
        m = phi.shape[0]
        return self.lambd * np.eye(m) * np.sqrt(self.d/m)


class UniformHyperspherePrior(object):
    # THE NEGATIVE LOG PRIOR AND GRADIENTS OF GAUSSIAN PRIOR
    def __init__(self, m):
        self.m = m
        self.zeros = np.zeros((m,))
        self.I = np.zeros((m,m))

    def lp(self, phi):
        return 0

    def grad(self, phi):
        return self.zeros

    def hess(self, phi):
        return self.I


class Posterior(object):
    # POSTERIOR, WHOSE LOGARITHM IS LOG PRIOR + LOG LIKELIHOOD
    def __init__(self, prior, likelihood, V, x):
        self.prior = prior
        self.likelihood = likelihood
        self.V = V
        self.x = x

    def nlp(self, phi):
        ret = self.prior.lp(phi) + self.likelihood.nll(self.V, phi, self.x)
        #print(ret)
        return ret

    def grad(self, phi):
        ret =  self.prior.grad(phi) + self.likelihood.grad(self.V, phi, self.x)
        return ret

    def hess(self, phi):
        return self.prior.hess(phi) + self.likelihood.hess(self.V, phi, self.x)


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
    """
    def __init__(self, d, m, l, norm_sq):
        self.d = d
        self.m = m
        self.l = l
        self.norm_sq = norm_sq

        # Setup likelihood and prior
        self.prior = UniformHyperspherePrior(m)
        #TODO: hardcoded Gaussian
        self.likelihood = ExpFam(lambda z: z**2/2, lambda z: z, lambda z: z,
                lambda z: np.ones_like(z))
        
        # Set up the kernel C and the matrix V
        self.C = KernelC(l, d)
        self.V = np.random.normal(0, 1/np.sqrt(m), (d, m))

    def _find_gamma(self, x):
        """ Map x to gamma """
        return self.C.features(x).reshape((-1,))

    def _find_psi(self, x):
        """ Solve m dimensional optimisation problem for psi"""
        gamma = self._find_gamma(x)
        # Define prior and likelihood
        lp = Posterior(self.prior, self.likelihood, self.V, gamma)

        # Define constraints on the hypersphere
        constraints = opt.NonlinearConstraint(\
            lambda x: np.linalg.norm(x)**2/self.m-self.norm_sq,
            0, 0, jac=lambda x: 2*x/self.m, hess = opt.BFGS())

        # Find the MAP
        p0 = np.random.normal(0, 1., (self.m,))
        psi = opt.minimize(lp.nlp, p0, jac=lp.grad, method='trust-constr', 
            hess=opt.BFGS(), constraints=[constraints])#, options={'disp':True})

        return psi.x

    def _find_limiting_kernel(self, x, xdash):
        """ Wrapper for the first type of kernel """
        c11 = self.C.kernel(x, x)
        c22 = self.C.kernel(xdash, xdash)
        c12 = self.C.kernel(x, xdash)

        lamb1 = np.sqrt(c11/self.norm_sq+1)
        lamb2 = np.sqrt(c22/self.norm_sq+1)

        limit_kernel11 = c11/(lamb1**2-1)
        limit_kernel22 = c22/(lamb2**2-1)
        limit_kernel12 = c12/(lamb1*lamb2-1)

        return limit_kernel11, limit_kernel22, limit_kernel12, lamb1, lamb2

    def find_limiting_kernel(self, x, xdash):
        """ The first type of kernel """
        return self._find_limiting_kernel(x, xdash)[2]

    def find_finite_kernel(self, x, xdash):
        """ The second type of kernel"""
        psi = self._find_psi(x)
        psidash = self._find_psi(xdash)
        return psi.T @ psidash/self.m

    def find_one_step_kernel(self, x, xdash):
        """ The third type of kernel""" 
        gamma = self._find_gamma(x)
        gammadash = self._find_gamma(xdash)
        limit_kernel11, limit_kernel22, limit_kernel12, lamb1, lamb2 = \
        self._find_limiting_kernel(x, xdash)

        rho = np.clip(limit_kernel12/np.sqrt(limit_kernel11*limit_kernel22), -1, 1)

        r = np.zeros((self.m,)); rdash = np.zeros((self.m,))
        r[0] = np.sqrt(limit_kernel11*m)
        rdash[0] = rho; rdash[1] = np.sqrt(1 - rho**2)
        rdash = rdash * np.sqrt(limit_kernel22*m)

        kd_feat_r = self.likelihood.grad(self.V, r, gamma)
        kd_feat_rdash = self.likelihood.grad(self.V, rdash, gammadash)
        kd_res = kd_feat_r.T@kd_feat_rdash/(lamb1*lamb2*self.d)

        return kd_res



if __name__ == '__main__':
    # Simulation Parameters
    d       = 300
    m       = int(d**(3/2)) # choose m >> d
    l       = 2
    norm_sq = 1**2


    x1 = np.linspace(-5, 5, 10)
    x2 = np.linspace(-5, 5, 10)
    xarr = np.transpose([np.tile(x1, len(x2)), np.repeat(x1, len(x2))])

    for x in xarr:
        for xdash in xarr:
            #print(measure_residual(x.reshape((-1,1)), xdash.reshape((-1,1)),
            #    d, m, l))
            x = x.reshape((-1,1))
            xdash = xdash.reshape((-1,1))
            K = KernelThreeTypes(d, m, l, norm_sq)
            print(K.find_limiting_kernel(x, xdash))
            print(K.find_finite_kernel(x, xdash))
            print(K.find_one_step_kernel(x, xdash))
            print('')

