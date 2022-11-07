import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import scipy.spatial as sp
import pandas as pd

def matplotlib_config():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 20
    matplotlib.rcParams['ytick.labelsize'] = 20
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

def plot_kernel_matrix(K, fname):
    plt.imshow(K)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def align_kernel_matrix(K):
    # https://www.jmlr.org/papers/volume13/cortes12a/cortes12a.pdf equation 1
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return Kc

def cka(K1, K2):
    # Centered kernel alignment 
    # https://www.jmlr.org/papers/volume13/cortes12a/cortes12a.pdf Definition 4
    # 0 is dissimilar, 1 is similar
    K1c = align_kernel_matrix(K1)
    K2c = align_kernel_matrix(K2)
    return np.sum(K1c * K2c) / (np.linalg.norm(K1c)*np.linalg.norm(K2c))

def frobenius_dist(K1, K2):
    return np.linalg.norm(K1 - K2)

def operator_dist(K1, K2):
    return np.linalg.norm(K1 - K2, ord=2)

def plot_err_versus_dimension(Kinfinite, Kfinitelist, dlist, metric):
    assert len(dlist) == len(Kfinitelist)
    err_list = []
    for K in Kfinitelist:
        alignment = metric(K, Kinfinite)
        err_list.append(alignment)

    dlist_, err_list_ = (list(t) for t in zip(*sorted(zip(dlist, err_list))))
    plt.plot(dlist_, err_list_, zorder=10)

def plot_err_versus_dimension_all(Kreflist, Kfinitelist, dlist, Nlist, 
    steplist, fname, metric_str=cka):
    if metric_str == 'cka':
        metric = cka
    elif metric_str == 'frob':
        metric = frobenius_dist
    elif metric_str == 'spec':
        metric = operator_dist

    for K in Kreflist:
        plot_err_versus_dimension(K, Kfinitelist, dlist, metric)
    plt.xlabel(r'$d$')
    if metric_str == 'cka':
        plt.ylabel('CKA')
        plt.ylim([-0.1,1.1])
        plt.plot([0, np.amax(dlist)], [1,1], 'r--')
    elif metric_str == 'frob':
        plt.ylabel(r'$\Vert \cdot \Vert_F$')
        plt.plot([0, np.amax(dlist)], [0,0], 'r--')
    elif metric_str == 'spec':
        plt.ylabel(r'$\Vert \cdot \Vert_2$')
        plt.plot([0, np.amax(dlist)], [0,0], 'r--')
    plt.tight_layout()
    plt.savefig(fname + '_d.png', bbox_inches='tight')
    plt.close()

    for K in Kreflist:
        plot_err_versus_dimension(K, Kfinitelist, Nlist, metric)
    plt.xlabel(r'Number SGD iterations')
    if metric_str == 'cka':
        plt.ylabel('CKA')
        plt.ylim([-0.1,1.1])
    elif metric_str == 'frob':
        plt.ylabel(r'$\Vert \cdot \Vert_F$')
    elif metric_str == 'spec':
        plt.ylabel(r'$\Vert \cdot \Vert_2$')
    plt.tight_layout()
    plt.savefig(fname + '_t.png', bbox_inches='tight')
    plt.close()

    for K in Kreflist:
        plot_err_versus_dimension(K, Kfinitelist, steplist, metric)
    plt.xlabel(r'$\alpha^{(t)}$')
    if metric_str == 'cka':
        plt.ylabel('CKA')
        plt.ylim([-0.1,1.1])
        plt.plot([0, 0.2], [1, 1], color='red', linestyle='dashed')
    elif metric_str == 'frob':
        plt.ylabel(r'$\Vert \cdot \Vert_F$')
        if PREFIX == 'small_step':
            plt.xlim([-0.001, 0.201])
            #plt.ylim([0, 8])
        plt.plot([0, 0.2], [0, 0], color='red', linestyle='dashed')
    elif metric_str == 'spec':
        plt.ylabel(r'$\Vert \cdot \Vert_2$')
        if PREFIX == 'small_step':
            plt.xlim([-0.001, 0.201])
            #plt.ylim([0, 6])
        plt.plot([0, 0.2], [0, 0], color='red', linestyle='dashed')
    if PREFIX == 'small_step':
        plt.plot([0, 0], [-0.1, 80], color='red', linestyle='dashed')
        plt.plot([0.125, 0.125], [-0.1, 80], color='red', linestyle='dashed')

    plt.tight_layout()
    plt.savefig(fname + '_step.png', bbox_inches='tight')
    plt.close()


def baseline_kernel(max_=1, size=10):
    x1 = np.linspace(-5, 5, size)
    x2 = np.linspace(-5, 5, size)
    X = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
    return max_*np.exp(-sp.distance.cdist(X, X, 'sqeuclidean')/2)


if __name__ == '__main__':
    DATA_DIR = 'outputs/change_d_relu_new/'
    OUTPUT_DIR = '/home/tsu007/dek/outputs/change_d_relu_new/'
    PREFIX = ''

    matplotlib_config()

    Kinfinite = np.load(DATA_DIR + 'kpsi_65_400_1.0.npy')
    Kbaseline = baseline_kernel(np.amax(Kinfinite), size=int(np.sqrt(Kinfinite.shape[0])))

    Klist = []
    dlist = []
    Nlist = []
    steplist = []

    for fname in os.listdir(DATA_DIR):
        bo = '400_1.0' in fname
        if ('kfinite' in fname and (PREFIX in fname)) and bo:
            K = np.load(DATA_DIR + fname)
            fname_ = fname.replace('_', ' ').replace('.npy', ' ')
            str_split = [float(s) for s in fname_.split(' ') if s.replace('.','',1).\
                replace('e-','',1).isdigit()]
            d = int(str_split[0])
            N = int(str_split[1])
            step = str_split[2]
            plot_kernel_matrix(K, OUTPUT_DIR + fname[:-3] + 'png')
            Klist.append(K)
            dlist.append(d)
            Nlist.append(N)
            steplist.append(step)
        else:
            continue
        
    plot_err_versus_dimension_all([Kinfinite, Kbaseline], 
        Klist, dlist, Nlist, steplist, OUTPUT_DIR + 'cka', metric_str='cka') 
    plot_err_versus_dimension_all([Kinfinite, Kbaseline], 
        Klist, dlist, Nlist, steplist, OUTPUT_DIR + 'frob', metric_str='frob') 
    plot_err_versus_dimension_all([Kinfinite, Kbaseline], 
        Klist, dlist, Nlist, steplist, OUTPUT_DIR + 'spec', metric_str='spec') 


        
