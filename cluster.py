import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def random_runs(obj_func, n_runs=10, test_iter=10):
    def wrapper(f):
        def new_f(X, *args, **kwargs):
            new_args = kwargs.copy()
            new_args['retAll'] = False; new_args['n_iter']=test_iter
            pars_list, obj_list = [], []
            for i in range(n_runs):
                pars, c = f(X, *args, **new_args)
                pars_list.append(pars)
                obj_list.append(obj_func(X, pars))
            kwargs['init_pars'] = pars_list[np.argmin(obj_list)]
            return f(X, *args, **kwargs)
        return new_f
    return wrapper

def k_means_obj(X, pars, get_clust=False):
    rep = pars; n = X.shape[0]
    c = np.array([np.argmin(np.linalg.norm(X[i].reshape(1, -1) - rep, axis=1)) for i in range(n)])
    p = np.mean([np.linalg.norm(X[i] - rep[c[i]]) ** 2 for i in range(n)])
    return (p,c) if get_clust else p

@random_runs(k_means_obj, n_runs=10, test_iter=10)
def kmeans(X, k, n_iter=10 ** 10, retAll = False, init_pars=None):
    n = X.shape[0]

    # Initialize representatives as random vectors,
    # and clusters based on these representatives
    c_old = np.zeros(n);
    if init_pars is None:
        rep = X[np.random.choice(n, size=k, replace=False), :]
    else:
        rep = init_pars

    p, c = k_means_obj(X, rep, get_clust=True)

    # Maybe save things
    if retAll:
        allC = [c]; allRep = [rep]; allP = [p];

    # Run the kmeans algorithm till the clustering converges
    for i in range(n_iter):
        if np.all(c == c_old):
            break

        c_old = c;

        # Maximization
        rep = np.array([np.mean(X[c == i, :], axis=0) for i in range(k)])

        # Expectation
        p, c = k_means_obj(X, rep, get_clust=True)

        # Maybe save things:
        if retAll:
            allC += [c]; allRep += [rep]; allP += [p];

    return (rep, c) + ((allRep, allC, allP) if retAll else ())

def GMMSamples(mus, sigmas, pi=None, nPoints=1000, retLatent = False):
    '''
    This will generate data from a guassian mixture model, given
    mus, a (nGaussians, D) shape ndarray that gives the centroid
    for each of the nGaussians components, as well as sigmas,
    a (nGaussians, D, D) shape ndarray that gives the cov matrix
    for each component.
    '''
    # Get data from all gaussians
    nGaussians, D = mus.shape
    assert(sigmas.shape[0] == nGaussians)
    allData = [np.random.multivariate_normal(mus[i], sigmas[i], size=nPoints) for i in range(nGaussians)]

    # Create weights

    weights = np.ones(nGaussians) / nGaussians if pi is None else pi
    assert(np.isclose(np.sum(weights), 1) and weights.shape[0] == nGaussians)

    # Choose data points from the different models based on weight
    data = np.empty((nPoints, D))
    inds = np.random.choice(range(nGaussians), size=nPoints, p=weights)
    for i, ind in enumerate(inds):
        data[i] = allData[ind][i]

    return (data, inds) if retLatent else data

def c_probs(X, mu, var, pi, normalize=True):
    n, d = X.shape; k = mu.shape[0];
    probs = np.empty((n, k))
    for i in range(k):
        probs[:, i] = st.multivariate_normal.pdf(X, mean=mu[i], cov=var[i])
    probs *= pi[None, :]
    if normalize:
        probs /= np.sum(probs, axis=1)[:, None]
    return probs

def pars(X, clust_probs, mu=None):
    n, d = X.shape; k = clust_probs.shape[1];
    if mu is None:
        mu = (clust_probs.T @ X) / np.sum(clust_probs, axis=0)[:, None]
    sum_probs = np.sum(clust_probs, axis=0)
    pi = sum_probs / n
    var = np.empty((k, d, d));
    for i in range(k):
        X2 = (X - mu[i].reshape(1, -1)) * clust_probs[:, i:i+1]
        var[i] = 1 / sum_probs[i] * (X2.T @ X2)
    return mu, var, pi

def neg_log_like(X, pars):
    mu, var, pi = pars
    probs = c_probs(X, mu, var, pi, normalize=False)
    return -np.sum(np.log(np.sum(probs, axis=1)))

@random_runs(neg_log_like, n_runs=20, test_iter=5)
def GMM(X, k, n_iter=5, retAll = False, init_pars=None):
    n, d = X.shape

    # Initialize means as random vectors sampled from X, initialize var and pi off that
    if init_pars is None:
        mu = X[np.random.choice(n, size=k, replace=False), :]
        c = np.array([np.argmin(np.linalg.norm(X[i].reshape(1, -1) - mu, axis=1)) for i in range(n)])
        clust_probs = np.arange(k) == c[:, None].astype(int)
        mu, var, pi = pars(X, clust_probs, mu)
    else:
        mu, var, pi = init_pars
        clust_probs = c_probs(X, mu, var, pi)
 
    p_old = neg_log_like(X, (mu, var, pi))
    
    # Maybe save things
    if retAll:
        allC = [clust_probs]; allRep = [mu]; allP = [p_old];
        
    # Run the kmeans algorithm till the clustering converges
    for i in range(n_iter):
        
        # Maximization
        mu, var, pi = pars(X, clust_probs)
        
        # Expectation
        clust_probs = c_probs(X, mu, var, pi)
    
        p = neg_log_like(X, (mu, var, pi))
        if p >= p_old:
            break
        p_old = p
        
        # Maybe save things:
        if retAll:
            allC += [clust_probs]; allRep += [mu]; allP += [p];
           
    return ((mu, var, pi), clust_probs) + ((allRep, allC, allP) if retAll else ())
