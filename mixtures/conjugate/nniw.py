import numpy as np
from scipy.special import gammaln
from scipy.stats import multivariate_normal as mvn, invwishart as iw

class nniw:
  def __init__(self, m, k, S, v):
    self.m = np.asarray(m, dtype=np.double)
    self.k = k
    self.S = np.asarray(S, dtype=np.double)
    self.v = v
    self.D = len(m)
    assert S.shape[0] == self.D and S.shape[1] == self.D, "S must be DxD"
    assert v >= self.D, 'v must be >= dimension D.'
    assert k > 0, 'k must be > 0'

  def posterior_sufficient_stats(self, Y):
    """ Compute the sufficient statistics of posterior p(theta | Y). """
    Y = self._process_observations(Y)
    N = Y.shape[0]
    if N == 0: return self.m, self.k, self.S, self.v

    yBar = np.mean(Y, axis=0)
    kN = self.k + N
    vN = self.v + N
    mN = (self.k * self.m + N*yBar) / kN
    SN = self.S + np.einsum('ij,ik->jk', Y, Y) + \
      self.k*np.outer(self.m, self.m) - \
      kN * np.outer(mN, mN)
    return mN, kN, SN, vN

  def posterior(self, Y):
    """ p(theta | Y) returned as nniw object. """
    return nniw(*self.posterior_sufficient_stats(Y))
  
  def update(self, Y):
    """ In-place conjugate posterior update, adding data Y. """
    self.m, self.k, self.S, self.v = self.posterior_sufficient_stats(Y)

  def downdate(self, Y):
    """ In-place downdate, removing data Y. """
    assert False, 'Not implemented.'

  def log_marginal_likelihood(self, Y):
    """ Compute p(Y; phi) = int_theta p(Y, theta; phi). """
    Y = self._process_observations(Y)
    N, D = Y.shape
    mN, kN, SN, vN = self.posterior_sufficient_stats(Y)

    t1 = -0.5*N*self.D*np.log(np.pi)
    t2 = 0.5*self.D * (np.log(self.k) - np.log(kN))
    t3 = 0.5 * (self.v*np.linalg.slogdet(self.S)[1] - \
      vN*np.linalg.slogdet(SN)[1])
    t4n = np.sum( gammaln(0.5 * (vN + 1 - np.arange(1, D+1))) )
    t4d = np.sum( gammaln(0.5 * (self.v + 1 - np.arange(1, D+1))) )
    return t1 + t2 + t3 + (t4n - t4d)

  def log_prior_predictive(self, Y):
    """ Return prior predictive distribution, p(y; phi). """
    Y = self._process_observations(Y)
    return self._logmvtpdf(Y, self.m, \
        (( self.k+1 )/( self.k * (self.v - self.D + 1) ))*self.S, \
        self.v - self.D + 1)

  def log_posterior_predictive(self, y, Y):
    """ Return posterior predictive distribution, p(y | Y; phi). """
    mN, kN, SN, vN = self.posterior_sufficient_stats(Y)
    return self._logmvtpdf(y, mN, (kN+1)/(kN*(vN-self.D+1))*SN, vN-self.D+1)

  def sample(self):
    """ Sample mu, Sigma from NIW(phi). """
    Sigma = iw.rvs(self.v, self.S)
    mu = mvn.rvs(self.m, (1/self.k)*Sigma)
    return mu, Sigma

  def sampleLogpdf(self, mu, Sigma):
    return iw.logpdf(Sigma, self.v, self.S) + \
      mvn.logpdf(mu, self.m, (1/self.k)*Sigma)

  def dataLogpdf(self, Y, mu, Sigma):
    return mvn.logpdf(Y, mu, Sigma)

  def _process_observations(self, Y):
    """ Enforce observations Y to be N x D. """
    Y = np.asarray(Y)
    if Y.ndim==1: Y = Y[np.newaxis,:]
    assert Y.shape[1] == self.D, 'Y must be N x D.'
    return Y

  def _logmvtpdf(self, Y, mu, Sigma, v):
    """ Compute log multivariate-t pdf for each y in Y. """
    Y = self._process_observations(Y)
    N, D = Y.shape

    t1n = gammaln(0.5 * (v + D))
    t1d1 = gammaln(0.5 * v)
    t1d2 = 0.5*D*np.log(v)
    t1d3 = 0.5*D*np.log(np.pi)
    t1d4 = 0.5*np.linalg.slogdet(Sigma)[1]

    yCtr = Y - mu
    SigmaI = np.linalg.inv(Sigma)
    mahal2 = np.einsum('nj,jk,nk->n', yCtr, SigmaI, yCtr)
    t2 = -0.5 * (v + self.D) * np.log(1 + (1.0/v)*mahal2)

    return t1n - (t1d1 + t1d2 + t1d3 + t1d4) + t2
