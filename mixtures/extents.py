from .conjugate import nniw, gem, dirmul, nniw_semi
import numpy as np
from .common import catrnd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn, invwishart as iw
from scipy.special import logsumexp
from tqdm import tqdm

def fit_extents_model(Y, K, **kwargs):
  """ Fit extents model to data Y.
  
  Fully-conjugate NIW parts prior, no target mean location updates for
  use in blocked gibbs sampling procedure.
  
  WARNING: Use fit_extents_model3, not this one.

  INPUT
    Y (ndarray, [N, D]): Observations
    K (int): Number of targets

  KEYWORDS
    (Model Priors)
    xPrior (ndarray, [K, D]): target location prior means
    Lambda (ndarray, [K, D, D]): target location prior covariances
    kappa (float): shared target extents prior mean-covariance tie
    S (ndarray, [D, D]): shared target extents covariance scatter prior
    v (float): shared target extents covariance concentration prior
    
    (Algorithm Parameters)
    nSamples (int): number of samples to draw
    maxBreaks (int): stick-breaking truncation parameter
    zInitStrategy (str): z initialization {'random', 'zeros'}
    deltaInitStrategy (str): delta initialization {'random', 'zeros', 'kmeans'}
    altLL (ndarray, [N,]): per-observation alternative LL
    sideLL (ndarray, [N, K]): per-target, per-observation side LL
    showProgress (bool): show sample progress bar

  OUTPUT
    z (ndarray, int, [nSamples, N]): target extent associations
    delta (ndarray, int, [nSamples, N]): target associations
    pi (ndarray, [nSamples, K, maxBreaks]): stick-breaking extent weights
    Pi (ndarray, [nSamples, K+nAlt]): target weights
    mu (ndarray, [nSamples, K, maxBreaks, D]): target extent means
    Sigma (ndarray, [nSamples, K, maxBreaks, D, D]): target extent covariances
    x (ndarray, [nSamples, K, D]): target locations
    ll (ndarray, [nSamples,]): joint sample log-likelihood
  """
  N, D = Y.shape

  # additional likelihood terms
  altLL = kwargs.get('altLL', None)
  if altLL is not None:
    assert altLL.ndim == 1 and len(altLL) == N, 'altLL bad shape'
    nAlt = 1
  else: nAlt = 0

  sideLL = kwargs.get('sideLL', None)
  if sideLL is not None:
    assert sideLL.ndim == 2 and sideLL.shape[0] == K and sideLL.shape[1] == N, \
      'sideLL bad shape'

  # algorithm parameters
  maxBreaks = kwargs.get('maxBreaks', 10)
  nSamples = kwargs.get('nSamples', 100)
  zInitStrategy = kwargs.get('zInitStrategy', 'random')
  deltaInitStrategy = kwargs.get('deltaInitStrategy', 'random')

  # model priors
  alpha = kwargs.get('alpha', 1.0)
  beta = kwargs.get('beta', np.ones(K+nAlt) / (K+nAlt))

  xPrior = kwargs.get('xPrior', np.tile(np.mean(Y, axis=0), (K,1)))
  Lambda = kwargs.get('Lambda', np.tile(np.cov(Y.T), (K,1,1)))

  kappa = kwargs.get('kappa', 1.0)
  S = kwargs.get('S', 0.25*np.cov(Y.T))
  v = kwargs.get('v', 5)

  # samples
  z = np.zeros((nSamples, N), dtype=np.int)
  delta = np.zeros((nSamples, N), dtype=np.int)
  pi = np.zeros((nSamples, K, maxBreaks)) # stick breaking extent weights
  Pi = np.zeros((nSamples, K+nAlt)) # dirichlet target weights
  mu = np.zeros((nSamples, K, maxBreaks, D))
  Sigma = np.zeros((nSamples, K, maxBreaks, D, D))
  x = np.zeros((nSamples, K, D))
  ll = np.zeros(nSamples)

  # initialize variables
  ## z
  if zInitStrategy == 'random': prevZ = np.random.randint(maxBreaks, size=N)
  elif zInitStrategy == 'zeros': prevZ = np.zeros(N, dtype=np.int)

  ## delta
  if deltaInitStrategy == 'random': prevDelta = np.random.randint(K+nAlt, size=N)
  elif deltaInitStrategy == 'zeros': prevDelta = np.zeros(N, dtype=np.int)
  elif deltaInitStrategy == 'kmeans': prevDelta = KMeans(K).fit(Y).labels_
  
  ## x
  prevX = np.stack( [mvn.rvs(xPrior[k], Lambda[k]) for k in range(K) ] )

  if kwargs.get('showProgress', False): sampleRange = tqdm(range(nSamples))
  else: sampleRange = range(nSamples)

  # begin sampling
  for s in sampleRange:
    # sample mu_{k,p}, Sigma_{k,p}, pi, Pi
    for k in range(K):
      nniwPrior = nniw(prevX[k], kappa, S, v)
      for d in range(maxBreaks):
        Ykd = Y[np.logical_and(prevDelta == k+nAlt, prevZ == d)]
        mu[s, k, d], Sigma[s, k, d] = nniwPrior.posterior(Ykd).sample()
        ll[s] += nniwPrior.sampleLogpdf(mu[s, k, d], Sigma[s, k, d])

      # update extent weights pi_k for each target
      zk = prevZ[ prevDelta == k+nAlt ] # these zk are internally consistent
      gemPrior = gem(alpha, maxBreaks=maxBreaks)
      pi[s,k] = gemPrior.posterior(zk).sample()
      ll[s] += gemPrior.sampleLogpdf(pi[s,k])

    dirmulPrior = dirmul(beta)
    Pi[s] = dirmulPrior.posterior(prevDelta).sample()
    ll[s] += dirmulPrior.sampleLogpdf(Pi[s])

    # sample x
    for k in range(K):
      # xMu = xPrior[k]
      xMu = prevX[k]
      xSig = Lambda[k]

      # perform measurement updates, if any
      for d in range(maxBreaks):
        if np.sum( np.logical_and(prevZ == d, prevDelta == k+nAlt) ) == 0:
          continue

        # mu[s,k,d] is observation d for target k (if it has associated obs)
        # (1/kappa)*Sigma[s,k,d] is observation d covariance for target k
        yInv = np.linalg.inv(kappa*Sigma[s, k, d])
        xInv = np.linalg.inv(xSig)
        xSig = np.linalg.inv(xInv + yInv)
        xMu = xSig.dot( yInv.dot(mu[s, k, d]) + xInv.dot(xMu) )
      x[s, k] = mvn.rvs(xMu, xSig)
    ll[s] += mvn.logpdf(x[s, k], xPrior[k], Lambda[k])
    
    # compute logPi, logpi safely, we'll need them shortly
    logPi = np.zeros(K+nAlt)
    for k in range(K+nAlt):
      if Pi[s,k] < 1e-100: logPi[k] = 1e-100
      else: logPi[k] = np.log(Pi[s,k])

    logpi = np.zeros((K, maxBreaks))
    for k in range(K):
      for d in range(maxBreaks):
        if pi[s,k,d] < 1e-100: logpi[k,d] = 1e-100
        else: logpi[k,d] = np.log(pi[s,k,d])
    # done computing logPi, logpi

    # sample delta
    targetLL = np.zeros((K+nAlt, N))
    if nAlt == 1: targetLL[0] = altLL
    if sideLL is not None: targetLL[1:] += sideLL

    for k in range(K):
      ll_k = np.zeros((maxBreaks, N))
      for d in range(maxBreaks):
        # log weight
        ll_k[d] += logpi[k, d]

        # log observation likelihood
        ll_k[d] += mvn.logpdf(Y, mu[s,k,d], Sigma[s,k,d])
      targetLL[k+nAlt] += logsumexp(ll_k, axis=0)
      targetLL[k+nAlt] += logPi[k+nAlt]
    pDelta = np.exp(targetLL - logsumexp(targetLL, axis=0, keepdims=True))
    delta[s] = catrnd(pDelta.T)
    ll[s] += np.sum(targetLL[delta[s], range(N)])

    # sample z
    for k in range(K):
      inds = (delta[s] == k+nAlt)
      Nk = np.sum(inds)
      Yk = Y[inds]
      logzk = np.zeros((maxBreaks, Nk))
      for d in range(maxBreaks):
        logzk += logpi[k, d]
        logzk[d] += mvn.logpdf(Yk, mu[s, k, d], Sigma[s, k, d])
      pzk = np.exp(logzk - logsumexp(logzk, axis=0, keepdims=True))
      z[s, inds] = catrnd(pzk.T)
      ll[s] += np.sum(logzk[z[s, inds], range(Nk)])

    prevZ = z[s]
    prevDelta = delta[s]
    prevX = x[s]

  return z, delta, pi, Pi, mu, Sigma, x, ll

def fit_extents_model_semi_conjugate(Y, K, **kwargs):
  """ Fit extents model to data Y.

  Conditionally-conjugate NIW parts prior, with target mean location updates.
  Could possibly be used in an iterative x_t | z_t, z_t | x_t update scheme, but
  details aren't currently worked out.

  INPUT
    Y (ndarray, [N, D]): Observations
    K (int): Number of targets

  KEYWORDS
    (Model Priors)
    xPrior (ndarray, [K, D]): target location prior means
    Lambda (ndarray, [K, D, D]): target location prior covariances
    Gamma (ndarray, [K, D, D]): target extents covariances
    S (ndarray, [D, D]): shared target extents covariance scatter prior
    v (float): shared target extents covariance concentration prior
    
    (Algorithm Parameters)
    nSamples (int): number of samples to draw
    maxBreaks (int): stick-breaking truncation parameter
    zInitStrategy (str): z initialization {'random', 'zeros'}
    deltaInitStrategy (str): delta initialization {'random', 'zeros', 'kmeans'}
    xInitStrategy (str): x initialization {'priorMean', 'sample', 'dataMean'}

    altLL (ndarray, [N,]): per-observation alternative LL
    sideLL (ndarray, [N, K]): per-target, per-observation side LL
    showProgress (bool): show sample progress bar

  OUTPUT
    z (ndarray, int, [nSamples, N]): target extent associations
    delta (ndarray, int, [nSamples, N]): target associations
    pi (ndarray, [nSamples, K, maxBreaks]): stick-breaking extent weights
    Pi (ndarray, [nSamples, K+nAlt]): target weights
    mu (ndarray, [nSamples, K, maxBreaks, D]): target extent means
    Sigma (ndarray, [nSamples, K, maxBreaks, D, D]): target extent covariances
    x (ndarray, [nSamples, K, D]): target locations
    ll (ndarray, [nSamples,]): joint sample log-likelihood
  """
  N, D = Y.shape

  # additional likelihood terms
  altLL = kwargs.get('altLL', None)
  if altLL is not None:
    assert altLL.ndim == 1 and len(altLL) == N, 'altLL bad shape'
    nAlt = 1
  else: nAlt = 0

  sideLL = kwargs.get('sideLL', None)
  if sideLL is not None:
    assert sideLL.ndim == 2 and sideLL.shape[0] == K and sideLL.shape[1] == N, \
      'sideLL bad shape'

  # algorithm parameters
  maxBreaks = kwargs.get('maxBreaks', 10)
  nSamples = kwargs.get('nSamples', 100)
  zInitStrategy = kwargs.get('zInitStrategy', 'zeros')
  deltaInitStrategy = kwargs.get('deltaInitStrategy', 'random')
  xInitStrategy = kwargs.get('xInitStrategy', 'priorMean')

  # model priors
  alpha = kwargs.get('alpha', 1.0)
  beta = kwargs.get('beta', np.ones(K+nAlt) / (K+nAlt))

  xPrior = kwargs.get('xPrior', np.tile(np.mean(Y, axis=0), (K,1)))
  Lambda = kwargs.get('Lambda', np.tile(np.cov(Y.T), (K,1,1)))
  Gamma = kwargs.get('Gamma', np.tile(0.25 * np.cov(Y.T), (K,1,1)))

  S = kwargs.get('S', 0.25*np.cov(Y.T))
  v = kwargs.get('v', 5)

  # samples
  z = np.zeros((nSamples, N), dtype=np.int)
  delta = np.zeros((nSamples, N), dtype=np.int)
  pi = np.zeros((nSamples, K, maxBreaks)) # stick breaking extent weights
  Pi = np.zeros((nSamples, K+nAlt)) # dirichlet target weights
  mu = np.zeros((nSamples, K, maxBreaks, D))
  Sigma = np.zeros((nSamples, K, maxBreaks, D, D))
  x = np.zeros((nSamples, K, D))
  ll = np.zeros(nSamples)

  # initialize variables
  ## z
  if zInitStrategy == 'random': prevZ = np.random.randint(maxBreaks, size=N)
  elif zInitStrategy == 'zeros': prevZ = np.zeros(N, dtype=np.int)

  ## delta
  if deltaInitStrategy == 'random': prevDelta = np.random.randint(K+nAlt, size=N)
  elif deltaInitStrategy == 'zeros': prevDelta = np.zeros(N, dtype=np.int)
  elif deltaInitStrategy == 'kmeans': prevDelta = KMeans(K).fit(Y).labels_
  
  ## x
  if xInitStrategy == 'priorMean':
    prevX = xPrior
  elif xInitStrategy == 'sample': 
    prevX = np.stack( [mvn.rvs(xPrior[k], Lambda[k]) for k in range(K) ] )
  elif xInitStrategy == 'dataMean':
    prevX = np.tile(np.mean(Y, axis=0)[np.newaxis,:], [K, 1])

  ## mu
  prevMu = np.tile(prevX[:,np.newaxis,:], [1, maxBreaks, 1])

  ## visualization
  if kwargs.get('showProgress', False): sampleRange = tqdm(range(nSamples))
  else: sampleRange = range(nSamples)

  # begin sampling
  for s in sampleRange:
    # sample mu_{k,p}, Sigma_{k,p}, pi, Pi
    for k in range(K):
      for d in range(maxBreaks):
        Ykd = Y[np.logical_and(prevDelta == k+nAlt, prevZ == d)]
        
        vN, SN = nniw_semi.iwUpdate(Ykd, prevMu[k, d], v, S)
        Sigma[s, k, d] = iw.rvs(vN, SN)
        ll[s] += iw.logpdf(Sigma[s, k, d], vN, SN)

        mN, VN = nniw_semi.normalUpdate(Ykd, Sigma[s, k, d],
          prevX[k], Gamma[k])
        mu[s, k, d] = mvn.rvs(mN, VN)
        ll[s] += mvn.logpdf(mu[s, k, d], mN, VN)

      # update extent weights pi_k for each target
      zk = prevZ[ prevDelta == k+nAlt ] # these zk are internally consistent
      gemPrior = gem(alpha, maxBreaks=maxBreaks)
      pi[s,k] = gemPrior.posterior(zk).sample()
      ll[s] += gemPrior.sampleLogpdf(pi[s,k])

    dirmulPrior = dirmul(beta)
    Pi[s] = dirmulPrior.posterior(prevDelta).sample()
    ll[s] += dirmulPrior.sampleLogpdf(Pi[s])

    # sample x
    for k in range(K):
      xMu = prevX[k]
      xSig = Lambda[k]

      # perform measurement updates, if any
      for d in range(maxBreaks):
        Nk = np.sum( np.logical_and(prevZ == d, prevDelta == k+nAlt) )
        if Nk == 0: continue
        xMu, xSig = nniw_semi.normalUpdate(mu[s,k,d], Gamma[k], xMu, xSig)
      x[s, k] = mvn.rvs(xMu, xSig)
    ll[s] += mvn.logpdf(x[s, k], xPrior[k], Lambda[k])
    
    # compute logPi, logpi safely, we'll need them shortly
    logPi = np.zeros(K+nAlt)
    for k in range(K+nAlt):
      if Pi[s,k] < 1e-100: logPi[k] = 1e-100
      else: logPi[k] = np.log(Pi[s,k])

    logpi = np.zeros((K, maxBreaks))
    for k in range(K):
      for d in range(maxBreaks):
        if pi[s,k,d] < 1e-100: logpi[k,d] = 1e-100
        else: logpi[k,d] = np.log(pi[s,k,d])
    # done computing logPi, logpi

    # sample delta
    targetLL = np.zeros((K+nAlt, N))
    if nAlt == 1: targetLL[0] = altLL
    if sideLL is not None: targetLL[1:] += sideLL

    for k in range(K):
      ll_k = np.zeros((maxBreaks, N))
      for d in range(maxBreaks):
        # log weight
        ll_k[d] += logpi[k, d]

        # log observation likelihood
        ll_k[d] += mvn.logpdf(Y, mu[s,k,d], Sigma[s,k,d])
      targetLL[k+nAlt] += logsumexp(ll_k, axis=0)
      targetLL[k+nAlt] += logPi[k+nAlt]
    pDelta = np.exp(targetLL - logsumexp(targetLL, axis=0, keepdims=True))
    delta[s] = catrnd(pDelta.T)
    ll[s] += np.sum(targetLL[delta[s], range(N)])

    # sample z
    for k in range(K):
      inds = (delta[s] == k+nAlt)
      Nk = np.sum(inds)
      Yk = Y[inds]
      logzk = np.zeros((maxBreaks, Nk))
      for d in range(maxBreaks):
        logzk += logpi[k, d]
        logzk[d] += mvn.logpdf(Yk, mu[s, k, d], Sigma[s, k, d])
      pzk = np.exp(logzk - logsumexp(logzk, axis=0, keepdims=True))
      z[s, inds] = catrnd(pzk.T)
      ll[s] += np.sum(logzk[z[s, inds], range(Nk)])

    prevZ = z[s]
    prevDelta = delta[s]
    prevX = x[s]
    prevMu = mu[s]

  return z, delta, pi, Pi, mu, Sigma, x, ll


def fit_extents_model3(Y, K, **kwargs):
  """ Fit extents model v3 to data Y.

  Conditionally-conjugate NIW parts prior, no target mean location updates for
  use in blocked gibbs sampling procedure.

  INPUT
    Y (ndarray, [N, D]): Observations
    K (int): Number of targets

  KEYWORDS
    (Model Priors)
    x (ndarray, [K, D]): sampled target locations
    # Lambda (ndarray, [K, D, D]): target location prior covariances
    Gamma (ndarray, [K, D, D]): target extents covariances
    S (ndarray, [D, D]): shared target extents covariance scatter prior
    v (float): shared target extents covariance concentration prior
    
    (Algorithm Parameters)
    nSamples (int): number of samples to draw
    maxBreaks (int): stick-breaking truncation parameter
    zInitStrategy (str): z initialization {'random', 'zeros'}
    deltaInitStrategy (str): delta initialization {'random', 'zeros', 'kmeans'}

    altLL (ndarray, [N,]): per-observation alternative LL
    sideLL (ndarray, [N, K]): per-target, per-observation side LL
    showProgress (bool): show sample progress bar

  OUTPUT
    z (ndarray, int, [nSamples, N]): target extent associations
    delta (ndarray, int, [nSamples, N]): target associations
    pi (ndarray, [nSamples, K, maxBreaks]): stick-breaking extent weights
    Pi (ndarray, [nSamples, K+nAlt]): target weights
    mu (ndarray, [nSamples, K, maxBreaks, D]): target extent means
    Sigma (ndarray, [nSamples, K, maxBreaks, D, D]): target extent covariances
    ll (ndarray, [nSamples,]): joint sample log-likelihood
  """
  N, D = Y.shape

  # additional likelihood terms
  altLL = kwargs.get('altLL', None)
  if altLL is not None:
    assert altLL.ndim == 1 and len(altLL) == N, 'altLL bad shape'
    nAlt = 1
  else: nAlt = 0

  sideLL = kwargs.get('sideLL', None)
  if sideLL is not None:
    assert sideLL.ndim == 2 and sideLL.shape[0] == K and sideLL.shape[1] == N, \
      'sideLL bad shape'

  # algorithm parameters
  maxBreaks = kwargs.get('maxBreaks', 10)
  nSamples = kwargs.get('nSamples', 100)
  zInitStrategy = kwargs.get('zInitStrategy', 'zeros')
  deltaInitStrategy = kwargs.get('deltaInitStrategy', 'random')

  # model priors
  alpha = kwargs.get('alpha', 1.0)
  beta = kwargs.get('beta', np.ones(K+nAlt) / (K+nAlt))

  x = kwargs.get('x', np.tile(np.mean(Y, axis=0), (K,1)))
  # Lambda = kwargs.get('Lambda', np.tile(np.cov(Y.T), (K,1,1)))
  Gamma = kwargs.get('Gamma', np.tile(0.25 * np.cov(Y.T), (K,1,1)))

  S = kwargs.get('S', 0.25*np.cov(Y.T))
  v = kwargs.get('v', 5)

  # samples
  z = np.zeros((nSamples, N), dtype=np.uint8)
  delta = np.zeros((nSamples, N), dtype=np.uint8)
  # z = np.zeros((nSamples, N), dtype=np.int)
  # delta = np.zeros((nSamples, N), dtype=np.int)
  pi = np.zeros((nSamples, K, maxBreaks)) # stick breaking extent weights
  Pi = np.zeros((nSamples, K+nAlt)) # dirichlet target weights
  mu = np.zeros((nSamples, K, maxBreaks, D))
  Sigma = np.zeros((nSamples, K, maxBreaks, D, D))
  ll = np.zeros(nSamples)

  # initialize variables
  ## z
  if zInitStrategy == 'random': prevZ = np.random.randint(maxBreaks, size=N)
  elif zInitStrategy == 'zeros': prevZ = np.zeros(N, dtype=np.int)

  ## delta
  if deltaInitStrategy == 'random': prevDelta = np.random.randint(K+nAlt, size=N)
  elif deltaInitStrategy == 'zeros': prevDelta = np.zeros(N, dtype=np.int)
  elif deltaInitStrategy == 'kmeans': prevDelta = KMeans(K).fit(Y).labels_
  
  ## mu
  prevMu = np.tile(x[:,np.newaxis,:], [1, maxBreaks, 1])

  ## visualization
  if kwargs.get('showProgress', False): sampleRange = tqdm(range(nSamples))
  else: sampleRange = range(nSamples)

  # begin sampling
  for s in sampleRange:
    # sample mu_{k,p}, Sigma_{k,p}, pi, Pi
    for k in range(K):
      for d in range(maxBreaks):
        Ykd = Y[np.logical_and(prevDelta == k+nAlt, prevZ == d)]
        
        vN, SN = nniw_semi.iwUpdate(Ykd, prevMu[k, d], v, S)
        Sigma[s, k, d] = iw.rvs(vN, SN)
        ll[s] += iw.logpdf(Sigma[s, k, d], vN, SN)

        mN, VN = nniw_semi.normalUpdate(Ykd, Sigma[s, k, d],
          x[k], Gamma[k])
        mu[s, k, d] = mvn.rvs(mN, VN)
        ll[s] += mvn.logpdf(mu[s, k, d], mN, VN)

      # update extent weights pi_k for each target
      zk = prevZ[ prevDelta == k+nAlt ] # these zk are internally consistent
      gemPrior = gem(alpha, maxBreaks=maxBreaks)
      pi[s,k] = gemPrior.posterior(zk).sample()
      ll[s] += gemPrior.sampleLogpdf(pi[s,k])

    dirmulPrior = dirmul(beta)
    Pi[s] = dirmulPrior.posterior(prevDelta).sample()
    ll[s] += dirmulPrior.sampleLogpdf(Pi[s])

    # compute logPi, logpi safely, we'll need them shortly
    logPi = np.zeros(K+nAlt)
    for k in range(K+nAlt):
      if Pi[s,k] < 1e-100: logPi[k] = 1e-100
      else: logPi[k] = np.log(Pi[s,k])

    logpi = np.zeros((K, maxBreaks))
    for k in range(K):
      for d in range(maxBreaks):
        if pi[s,k,d] < 1e-100: logpi[k,d] = 1e-100
        else: logpi[k,d] = np.log(pi[s,k,d])
    # done computing logPi, logpi

    # sample delta
    targetLL = np.zeros((K+nAlt, N))
    if nAlt == 1: targetLL[0] = altLL
    if sideLL is not None: targetLL[1:] += sideLL

    for k in range(K):
      ll_k = np.zeros((maxBreaks, N))
      for d in range(maxBreaks):
        # log weight
        ll_k[d] += logpi[k, d]

        # log observation likelihood
        ll_k[d] += mvn.logpdf(Y, mu[s,k,d], Sigma[s,k,d])
      targetLL[k+nAlt] += logsumexp(ll_k, axis=0)
      targetLL[k+nAlt] += logPi[k+nAlt]
    pDelta = np.exp(targetLL - logsumexp(targetLL, axis=0, keepdims=True))
    delta[s] = catrnd(pDelta.T)
    ll[s] += np.sum(targetLL[delta[s], range(N)])

    # sample z
    for k in range(K):
      inds = (delta[s] == k+nAlt)
      Nk = np.sum(inds)
      Yk = Y[inds]
      logzk = np.zeros((maxBreaks, Nk))
      for d in range(maxBreaks):
        logzk += logpi[k, d]
        logzk[d] += mvn.logpdf(Yk, mu[s, k, d], Sigma[s, k, d])
      pzk = np.exp(logzk - logsumexp(logzk, axis=0, keepdims=True))
      z[s, inds] = catrnd(pzk.T)
      ll[s] += np.sum(logzk[z[s, inds], range(Nk)])

    prevZ = z[s]
    prevDelta = delta[s]
    prevMu = mu[s]

  return z, delta, pi, Pi, mu, Sigma, ll
