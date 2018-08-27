import numpy as np
from scipy.special import gammaln
from scipy.stats import multivariate_normal as mvn, invwishart as iw

def iwUpdate(y, mu, v0, S0):
  """ Return p(Sigma | v0, S0, y, mu) propto p(Sigma | v0, S0) p(y | mu, Sigma)

  INPUT
    y (ndarray, [N,D]): observations
    mu (ndarray, [D,]): observation mean estimate
    v0 (float): prior concentration for Sigma
    S0 (ndarray, [D,D]): prior scatter for Sigma

  OUTPUT
    vN (float): posterior inverse wishart concentration for Sigma
    SN (ndarray, [D,D]): posterior inverse wishart scatter for Sigma
  """
  y = _process_observations(y)
  N, D = y.shape
  assert mu.ndim == 1 and len(mu) == D, 'mu must be D-dimensional'
  assert v0 >= D, 'v0 must be >= D'
  assert S0.ndim == 2 and S0.shape[0] == S0.shape[1] == D, \
    'S0 must be D x D covariance'
  if N==0: return v0, S0

  y_mu = y - mu
  S_mu = np.einsum('ij,ik->jk', y_mu, y_mu)

  vN = v0 + N
  SN = S0 + S_mu
  return vN, SN

def normalUpdate(y, Sigma, m0, V0):
  """ Return p(mu | m0, V0, y, Sigma) propto p(mu | m0, V0) p(y | mu, Sigma)

  INPUT
    y (ndarray, [N,D]): observations
    Sigma (ndarray, [D,D]): estimated observation covariance
    m0 (ndarray, [D,]): prior mean for mu
    V0 (ndarray, [D,D]): prior covariance for mu

  OUTPUT
    mN (ndarray, [D,]): posterior mean for mu
    VN (ndarray, [D,D]): posterior covariance for mu
  """
  y = _process_observations(y)
  N, D = y.shape
  assert Sigma.ndim == 2 and Sigma.shape[0] == Sigma.shape[1] == D, \
    'Sigma must be D x D covariance'
  assert V0.ndim == 2 and V0.shape[0] == V0.shape[1] == D, \
    'V0 must be D x D covariance'
  assert m0.ndim == 1 and len(m0) == D, 'm0 must be D-dimensional'
  if N==0: return m0, V0

  yBar = np.mean(y, axis=0)
  V0i = np.linalg.inv(V0)
  SigmaI = np.linalg.inv(Sigma)
  
  VN = np.linalg.inv(V0i + N*SigmaI)
  mN = VN.dot( V0i.dot(m0) + N*SigmaI.dot(yBar) )
  return mN, VN

def _process_observations(Y):
  """ Enforce observations Y to be N x D. """
  Y = np.asarray(Y)
  if Y.ndim==1: Y = Y[np.newaxis,:]
  assert Y.ndim==2, 'Y must be N x D'
  return Y

if __name__ == "__main__":
  import matplotlib.pyplot as plt, du, du.stats
  N, D = (1000, 2)

  # sample parameters from model
  m0 = np.zeros(D)
  V0 = 10*np.eye(D)
  v0 = 5
  S0 = 15*np.eye(D)

  mu = mvn.rvs(m0, V0)
  Sigma = iw.rvs(v0, S0)
  y = mvn.rvs(mu, Sigma, size=N)

  # marginal posteriors
  mN, VN = normalUpdate(y, Sigma, m0, V0)
  mu_sample = mvn.rvs(mN, VN)
  vN, SN = iwUpdate(y, mu_sample, v0, S0)

  plt.scatter(y[:,0], y[:,1], s=1, c='k')
  plt.scatter(mu[0], mu[1], c='b', s=50)
  plt.plot(*du.stats.Gauss2DPoints(mu, Sigma), c='b', linestyle='--')
  plt.plot(*du.stats.Gauss2DPoints(mN, VN), c='r')
  plt.plot(*du.stats.Gauss2DPoints(mu_sample, SN/(vN-D-1)),
    c='r', linestyle='--')
  plt.legend(['true mu/Sigma', 'posterior mean dist',
    'mean posterior covariance', 'data', 'true mu'])
  plt.xlim(-10, 10)
  plt.ylim(-10, 10)
  plt.show()
