import numpy as np
from .common import catrnd, process_observations
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn
from .conjugate import nniw, dirmul

def gmm_logpdf_marginalized(Y, theta, pi):
  Y = process_observations(Y)
  N = Y.shape[0]
  K = len(pi)

  ll = np.stack([mvn.logpdf(Y, *theta[k]) for k in range(K)])
  logpi = np.array([1e-100 if pk<1e-100 else np.log(pk) for pk in pi])
  ll += logpi[:,np.newaxis]

  return logsumexp(ll, axis=0)

def gmm_fit_blocked_gibbs(Y, dirmul, phi, **kwargs):
  """ Fit GMM to data Y.

  INPUTS
    Y (ndarray, [N, D]): Observations
    dirmul (dirmul class): Object with the following methods:
      update, posterior, log_posterior_predictive, sample
    phi (list of nniw classes): Each must have the following methods:
      update, posterior, log_posterior_predictive, sample
  """
  nSamples = kwargs.get('nSamples', 100)
  N, D = Y.shape
  K = len(dirmul.alpha)

  # collection of parameter samples
  z = np.zeros((nSamples, N), dtype=np.int)
  pi = np.zeros((nSamples, K))
  theta = [ [] for n in range(nSamples) ]
  ll = np.zeros(nSamples)

  # randomly initialize labels z
  prevZ = np.random.randint(K, size=N)

  for s in range(nSamples):
    # sample mu, Sigma, pi given assignments z
    theta[s] = [ phi[k].posterior(Y[prevZ==k]).sample() for k in range(K) ]
    pi[s] = dirmul.posterior(prevZ).sample()

    # compute logpi safely, we'll need shortly
    logpi = np.zeros(K)
    for k in range(K):
      if pi[s,k] < 1e-100: logpi[k] = 1e-100
      else: logpi[k] = np.log(pi[s,k])

    # compute likelihood of each Y under all theta
    logz = np.zeros((N, K))
    for k in range(K):
      logz[:,k] = phi[k].dataLogpdf(Y, *theta[s][k]) + logpi[k]

    # sample new labels
    pz = np.exp(logz - logsumexp(logz, axis=1, keepdims=True))
    z[s] = catrnd(pz)

    # compute joint log-likelihood
    ll[s] += np.sum([phi[k].sampleLogpdf(*theta[s][k]) for k in range(K) ])
    ll[s] += (dirmul.sampleLogpdf(pi[s]) + np.sum(logz[range(N), z[s]]))

    prevZ = z[s]

  return z, pi, theta, ll
