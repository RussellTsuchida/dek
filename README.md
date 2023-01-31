# Reproducing the results of the paper

The main purpose of this code is to implement the a Deep Equilibrium Kernel discussed in the paper, and also to reproduce the results of Section 4.1 and 4.2.

## Reproducing the results in Section 4.1
Run the script `finite_dimensional.py`. The first argument specifies the dimension $d$, the second argument specifies the number of iterations to run SGD for. The third argment should be set to `1`.
In order to reproduce the results, you will need to run this script for different values of $d.$ We used values of $d$ between $5$ and $500$ in steps of $5$.

## Reproducing the results in Section 4.2
Run the script `evaluate_models.py`. The first argument is the type of kernel, which should be `dek`, `ntk`, `nnk`, or `rbf`. 
The second argument is the dataset, which should be `yacht`, `diabetes`, `energy1`, `energy2`, `concrete` or `wine`.
The last argument is a random seed, which can be any integer.
We ran this script for all seeds between $0$ and $99$, for every combination of the kernel and dataset.


## Deep Equilibrium Kernel
The `DEK` class is defined in `kernels.py`. This is a DEK that uses ReLU for $\sigma$, Heaviside step functions for $\rho$ and the arc-cosine kernel for $c$.
This class is instantiated with several hyperparameters:
`sigma_str` - Should always be set to 'relu'. Others are not implemented. As per $\sigma$ in the paper.
`rho_str` - Should always be set to 'heavy'. Others are not implemented. As per $\rho$ in the paper.
`C` - A PSD Kernel. Preimplemented options are 'relu'. As per $c$ in the paper.
`lambd` - Regularisation parameter. As per $\lambda$ in the paper.
`T` - The number of iterations to run SGD for (or equivalently, the number of function compositions of the DEK). If set to None, run a fixed point solver (Anderson acceleration) instead.
`mean_fun` - As per $\mu$ in the paper. kernels.ReLUMean() is implemented.
`c_scale` - A scaling factor for the kernel $c$.



