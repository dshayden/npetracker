import numpy as np
from scipy.special import gammaln
from scipy.stats import dirichlet

class dirmul:
  def __init__(self, alpha, oneHot=False):
    self.alpha = np.asarray(alpha)
    self.oneHot = oneHot
    self.K = len(alpha)

  def posterior_sufficient_stats(self, Y):
    """ Compute the sufficient statistics of posterior p(alpha | Y). """
    Y, Nk = self._process_observations(Y)
    return self.alpha + Nk

  def posterior(self, Y):
    """ p(theta | Y) returned as dirmul object. """
    return dirmul(self.posterior_sufficient_stats(Y), self.oneHot)
  
  def update(self, Y):
    """ In-place conjugate posterior update, adding data Y. """
    self.alpha = self.posterior_sufficient_stats(Y)

  def downdate(self, Y):
    """ In-place downdate, removing data Y. """
    assert False, 'Not implemented.'

  def sampleLogpdf(self, pi):
    """ log-likelihood of pi ~ dirmul.sample(). """
    return dirichlet.logpdf(pi, self.alpha)

  def log_marginal_likelihood(self, Y):
    """ Compute p(Y; phi) = int_pi p(Y, theta; alpha). """
    alphaN = self.posterior_sufficient_stats(Y)
    _, Nk = self._process_observations(Y)

    n = np.sum(Nk)

    t1 = gammaln(n+1) - np.sum(gammaln(Nk+1))
    t2 = log_multi_beta(alphaN) - log_multi_beta(self.alpha)

    # # test formula 1
    # alpha0 = np.sum(self.alpha)
    # _t1 = gammaln(n+1) + gammaln(alpha0) - gammaln(n + alpha0)
    # _t2 = np.sum(gammaln(Nk + self.alpha) - (gammaln(Nk+1) +
    #   gammaln(self.alpha)))
    #
    # # test formula 2
    # __t1n = np.log(n) + log_multi_beta([alpha0, n])
    # __t1d = np.sum( [ np.log(Nk[k]) + log_multi_beta([self.alpha[k], Nk[k]])
    #   for k in range(self.K) ] )
    #
    # assert np.allclose(t1+t2, _t1 + _t2), 'log_marginal_likelihood bad'
    # assert np.allclose(t1+t2, __t1n - __t1d), 'log_marginal_likelihood bad'

    return t1 + t2

  def log_posterior_predictive(self, y, Y):
    """ Return posterior predictive distribution, p(y | Y; phi). """
    Y, NkY = self._process_observations(Y)
    y, Nky = self._process_observations(y)

    t1 = gammaln(np.sum(Nky)+1) - np.sum(gammaln(Nky+1))
    t2 = log_multi_beta(Nky + NkY + self.alpha) - log_multi_beta(self.alpha)
    return t1 + t2

  def sample(self):
    """ Sample pi from Dir(alpha). """
    # return dirichlet.rvs(self.alpha)
    
    # todo: check if this breaks stuff
    return np.squeeze(dirichlet.rvs(self.alpha))

  def _process_observations(self, Y):
    """ Enforce observations Y to be N x D if oneHot else N. """
    Y = np.asarray(Y)
    if self.oneHot:
      if Y.ndim==1: Y = Y[np.newaxis,:]
      assert Y.shape[1] == self.K, 'Y must be N x K.'
      Nk = np.sum(Y, axis=0)
    else:
      Nk = np.zeros(self.K)
      vals, counts = np.unique(Y, return_counts=True)
      for idx, val in enumerate(vals):
        Nk[int(val)] = counts[idx]
      assert Y.ndim==1 or Y.ndim==0, 'Y must be N-dimensional.'
    return Y, Nk

def log_multi_beta(alpha):
  """ Log multivariate beta function. 
    
    Computes:
      log frac{ prod_k gamma(alpha_k) }{ gamma(sum_k alpha_k) }
  """
  return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
