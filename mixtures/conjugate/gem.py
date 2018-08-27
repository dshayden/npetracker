import numpy as np, matplotlib.pyplot as plt
import scipy.stats as ss
import IPython as ip

class gem:
  def __init__(self, *args, **kwargs):
    """ Instantiate GEM(alpha) distribution.

    INPUT (call 1)
      alpha (float, >0): concentration parameter

    INPUT (call 2)
      a (ndarray, floats, >0): GD parameter for each stick break
      b (ndarray, floats, >0): GD parameter for each stick break

    KEYWORD ARGS
      maxBreaks (int, >0): truncation level, len(a) == len(b) == maxBreaks
    """
    assert len(args)==1 or len(args)==2, 'must be 1 or 2 args.'

    if len(args) == 1:
      # call 1
      alpha = args[0]
      self.maxBreaks = kwargs.get('maxBreaks', 10)
      self.a = np.ones(self.maxBreaks)
      self.b = alpha*np.ones(self.maxBreaks)
    elif len(args) == 2:
      # call 2
      self.a = args[0]
      self.b = args[1]
      self.maxBreaks = len(self.a)
      assert len(self.a) == len(self.b), 'a, b must have same length'
    self.oneHot = kwargs.get('oneHot', False)

  def posterior_sufficient_stats(self, Y):
    Y, Nk = self._process_observations(Y)
    a = self.a + Nk
    b = self.b + (np.cumsum(Nk[::-1]) - Nk[::-1])[::-1]
    return a, b

  def posterior(self, Y):
    return gem(*self.posterior_sufficient_stats(Y), oneHot=self.oneHot)

  def update(self, Y):
    self.a, self.b = self.posterior_sufficient_stats(Y)

  def sampleLogpdf(self, pi):
    """ Log-likelihood of pi ~ gem(alpha). """
    # recover betas and evaluate log-likelihood
    assert len(pi) == self.maxBreaks, 'len(pi) != maxBreaks'
    K = self.maxBreaks
    beta = np.zeros(K)
    beta[0] = pi[0]

    if beta[0] >= 1: beta[0] = 1 - 1e-8
    ll = ss.beta.logpdf(beta[0], self.a[0], self.b[0])

    for k in range(1,K-1):
      beta[k] = pi[k] / np.prod(1 - beta[:k])
      if beta[k] >= 1: beta[k] = 1 - 1e-8
      if beta[k] <= 1e-8: beta[k] = 1e-8
      # if beta[k] <= 1e-8: beta[k] += 1e-8

      assert beta[k] > 0 and beta[k] < 1, 'beta bad'

      # if (1 - beta[k]) < 1e-8: beta[k] -= 1e-8
      # elif beta[k] <= 1e-8: beta[k] += 1e-8
      bpdf = ss.beta.logpdf(beta[k], self.a[k], self.b[k])
      assert not np.isinf(bpdf), 'INF'
      ll += bpdf
    return ll

  def sample(self):
    """ Sample pi from GEM(alpha). """
    beta = np.zeros(self.maxBreaks)
    pi = np.zeros(self.maxBreaks)

    for k in range(self.maxBreaks-1):
      beta[k] = ss.beta.rvs(self.a[k], self.b[k])
      pi[k] = beta[k] * np.prod(1-beta[:k])
    pi[-1] = 1 - np.sum(pi[:-1])
    return pi

  def _process_observations(self, Y):
    """ Enforce observations Y to be N x D if oneHot else N. """
    Y = np.asarray(Y)
    if self.oneHot:
      if Y.ndim==1: Y = Y[np.newaxis,:]
      assert Y.shape[1] == self.K, 'Y must be N x K.'
      Nk = np.sum(Y, axis=0)
    else:
      K = self.maxBreaks
      Nk = np.zeros(K)
      vals, counts = np.unique(Y, return_counts=True)
      for idx, val in enumerate(vals):
        Nk[int(val)] = counts[idx]
      assert Y.ndim==1 or Y.ndim==0, 'Y must be N-dimensional.'
    return Y, Nk

if __name__ == "__main__":
  np.set_printoptions(precision=4, suppress=True)

  maxBreaks = 20
  alpha = 0.01
  g = gem(alpha, maxBreaks)
  
  nTest = 1000
  for n in range(nTest):
    pi = g.sample()
    assert np.abs(np.sum(pi) - 1) < 1e-16, 'bad sample'
  print('sample test past')

  for n in range(nTest):
    pi = g.sample()
    ll = g.sampleLogpdf(pi)
    assert not np.isinf(ll)
    assert not np.isnan(ll)
    print('ll: %.8f' % ll)
