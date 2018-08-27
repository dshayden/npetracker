import numpy as np, argparse
import mixtures
import functools
import du, du.stats

import matplotlib.pyplot as plt
import IPython as ip

def build_opts(nY, nX, K, **kwargs):
  o = argparse.Namespace()

	# Parameters
  o.nY = nY
  o.nX = nX
  o.N = nY*nX
  o.K = K

  o.camK = kwargs.get('camK', None)
  o.camKi = np.linalg.inv(o.camK) if o.camK is not None else None
  o.Dy_loc = 3 if o.camK is not None else 2
  o.Dy_app = 3
  o.Dy = o.Dy_loc + o.Dy_app
  o.Dx = 2 * o.Dy_loc

  zer = np.zeros((o.Dy_loc, o.Dy_loc))
  eye = np.eye(o.Dy_loc)
  o.F = kwargs.get('F', np.block([[eye, eye], [zer, eye]]))
  o.H = kwargs.get('H', np.block([eye, zer]))

  o.lab = kwargs.get('lab', True)
  o.fgMask = kwargs.get('fgMask', np.ones((o.nY, o.nX), dtype=np.bool))
  o.fgMask = o.fgMask.flatten()

  o.nPartSamples = kwargs.get('nPartSamples', 100)
  # nPostSamples
  
  o.maxBreaks = kwargs.get('maxBreaks', 5)
  o.drawColors = kwargs.get('drawColors', du.diffcolors(K))

  # Convenience
  yy, xx = np.meshgrid(range(o.nY), range(o.nX), indexing='ij')
  o._xf, o._yf = (xx.flatten(), yy.flatten())
  o._xy = np.stack((o._xf, o._yf), axis=1).astype(np.float)

  # Priors
  ## parts
  o.Gamma = kwargs.get('Gamma', np.tile(np.eye(o.Dy_loc), (K, 1, 1)))
  o.v = kwargs.get('v', 2000)
  defaultS_scale = 50 if o.Dy_loc == 3 else 20
  o.S = kwargs.get('S', defaultS_scale*(o.v - o.Dy_loc - 1) * np.eye(o.Dy_loc))
  o.alpha = kwargs.get('alpha', 0.1)
  o.beta = kwargs.get('beta', np.array([0.98, 1, 1]))

  o.x0 = kwargs.get('x0', np.zeros((o.K, o.Dx)))
  o.P0 = kwargs.get('P0', np.tile(1e6*np.eye(o.Dx), [o.K, 1, 1]))
  
  # todo: integrate these when I add inference for them
  o.Lambda = kwargs.get('Lambda',
    np.tile(np.block([[1e-4*eye, zer], [zer, 20*eye]]), [K, 1, 1]))

  # [m_{nB}]_{n=1}^N, kappa_B, S_B, v_B
  # gamma
  # m_A, kappa_A, S_A, v_A
  return o

class BGModel:
  """ Observation background model. """

  def __init__(self, mu_B, Sigma_B, m):
    """ Per-observation background model.

    INPUT
      mu_B (ndarray, [N, 3+dy]): mean background
      Sigma_B (ndarray, [N, 3+dy]): (diagonal) background variances
      m (ndarray, [N,]): boolean mask indicating missing or non-missing bg model
    """
    self.mu_B = mu_B
    self.Sigma_B = Sigma_B
    self.m = m

  def logpdf(self, y):
    """ Calculate per-observation background log-likelihood.

    INPUT
      y (ndarray, [N, 3+dy]): observations

    OUTPUT
      ll (ndarray, [N,]): log-likelihood of background
    """
    return du.stats.logmvnpdfdiag(y, self.mu_B, self.Sigma_B)

class AppModel:
  """ Target appearance model. """

  def __init__(self, pi, mu_A, Sigma_A):
    """ Target appearance model.

    INPUT
      pi (ndarray, [C,]): appearance mixture coefficients
      mu_A (ndarray, [C, 3]): mean appearances
      Sigma_A (ndarray, [C, 3, 3]): appearance covariances 
    """
    self.C = len(pi)
    self.pi = pi
    self.mu_A = mu_A
    self.Sigma_A = Sigma_A

  def logpdf(self, yA):
    """ Calculate per-observation background log-likelihood.

    INPUT
      yA (ndarray, [N, 3]): appearance observations

    OUTPUT
      ll (ndarray, [N,]): log-likelihood of appearance 
    """
    theta = [ (self.mu_A[c], self.Sigma_A[c]) for c in range(self.C) ]
    return mixtures.gmm_logpdf_marginalized(yA, theta, self.pi)

def build_sample(o, x, Pf, mu, Sigma, Pi, pi, ll, **kwargs):
  # todo: check everything is expected format/type
  delta = kwargs.get('delta', None)
  z = kwargs.get('z', None)

  s = argparse.Namespace()
  s.x = x
  s.Pf = Pf
  s.mu = mu
  s.Sigma = Sigma
  s.Pi = Pi
  s.pi = pi
  s.ll = ll

  if delta is not None and z is not None:
    s.delta = delta
    s.z = z
  
  return s

# def build_outer_sample(**kwargs):
#   None
#   # x_{kL}^t
#
#   ## app
#   # pi_{kA}
#   # mu_{kcA}, Sigma_{kcA}
#
#   ## parts
#   # theta_{kp}^t
#   # pi_k^t
#   # Pi^t
#
#   ## background
#   # mu_{nB}, Sigma_{nB}
#
#   ## associations, store or not???
#   # delta_n^t
#   # z_n^t

def LoadObs(o, rgb, depth=None):
  """ Load registered (rgb, depth) image pair as observation.

  INPUT
    o (opts)
    rgb (str or ndarray): path name or already-loaded image
    depth (str or ndarray): path name or already-loaded image

  OUTPUT
    y (ndarray, [N, dy+3]): observations
    m (ndarray, [N,]): boolean mask, False at any missing observation
  """
  if type(rgb) == str: rgb = du.imread(rgb)  
  assert rgb.shape[0] == o.nY and rgb.shape[1] == o.nX, 'Invalid image size.' 

  if o.lab: img = du.rgb2labImg(rgb).astype(np.float)
  else: img = rgb.astype(np.float)
  m = np.ones(o.N, dtype=np.bool)

  if depth is not None and o.camK is not None:
    if type(depth) == str: depth = du.imread(depth)
    # any 
    xyZ = np.concatenate((o._xy,
      depth.astype(np.float).flatten()[:,np.newaxis]), axis=1)
    xyZ[:,:2] *= xyZ[:,2][:,np.newaxis]
    pts = xyZ.dot(o.camKi)
    m = depth.flatten() > 0
  else:
    pts = o._xy

  imgF = img.astype(np.float)
  imgF.shape = (o.N, 3)
  y = np.concatenate((pts, imgF), axis=1)

  return y, m

def SampleParts_t(o, x, bg, app, img, depth=None):
  """ Infer parts for a given time t.
  
  INPUT
    o (opts)
    rgb (str or ndarray): path name or already-loaded image
    depth (str or ndarray): path name or already-loaded image
    x (ndarray, [K, Dx]): sampled target locations
    bg (bgModel): background model
    app (length-K list of appModel): target appearance models
  """
  y, m = LoadObs(o, img, depth)
  
  # handle fgMask and missing data in observations + bgModel 
  useIdx = np.logical_and(m, np.logical_and(bg.m, o.fgMask))
  yUse = y[useIdx, :o.Dy_loc]

  # evaluate bg and appearance likelihoods
  bgLL = bg.logpdf(y)[useIdx]
  appLL = np.stack( 
    [app[k].logpdf(y[useIdx,o.Dy_loc:]) for k in range(o.K)])

  Hx = np.stack( [o.H.dot(x[k]) for k in range(o.K) ] )

  fit_kwargs = dict(
    nSamples=o.nPartSamples, altLL=bgLL, sideLL=appLL,
    alpha=o.alpha, v=o.v, S=o.S, Gamma=o.Gamma, x=Hx,
    maxBreaks=o.maxBreaks,
    zInitStrategy='random')
  res = mixtures.fit_extents_model3(yUse, o.K, **fit_kwargs)
  z, delta, pi, Pi, mu, Sigma, ll = res

  _z = 255*np.ones(o.N, dtype=np.uint8)
  _z[useIdx] = z[-1]
  _delta = 255*np.ones(o.N, dtype=np.uint8)
  _delta[useIdx] = delta[-1]

  _z, _pi, _mu, _Sigma = CollectUsedParts(
    o, _z, _delta, pi[-1], mu[-1], Sigma[-1])

  # return last joint sample 
  return _z, _delta, _pi, Pi[-1], _mu, _Sigma, ll[-1]

def SampleLocs(o, delta, z, mu, Sigma, x0, P0):
  """ Sample from latent posterior of linear dynamical system.

  Return sample from x_{1:T} ~ p(X | Delta, Z, Mu, Sigma, x0, P0) where
    X = x_{1:T}
    Delta = delta_{1:N}^{1:T}
    Z = z_{1:K,1:N}^{1:T}
    Mu = mu_{1:K,1:P_k}^{1:T}
    Sigma = Sigma_{1:K,1:P_k}^{1:T}

  INPUT
    o (opts):
    delta ( [delta^1, ..., delta^T] ): Target associations for time t
      delta^t (ndarray, [o.N,])
    z ( [z^1, ..., z^T] ): Part associations for time t
      z^t (ndarray, [o.N,])
    mu ( [mu^1, ..., mu^T] ): Target part means for time t
      mu^t ( [mu_1^t, ..., mu_K^t] )
        mu_k^t (ndarray, [nParts_k, o.Dy_loc])
    Sigma ( [Sigma^1, ..., Sigma^T] ): Target part covariances for time t
      Sigma^t ( [Sigma_1^t, ..., Sigma_K^t] )
        Sigma_k^t (ndarray, [nParts_k, o.Dy_loc, o.Dy_loc])
    x0 (ndarray, [o.K, o.Dx,]): Target prior locations at time t = -1
    P0 (ndarray, [o.K, o.Dx, o.Dx]): Target prior location covariance
      at time t = -1

  OUTPUT
    x (ndarray, [T, o.K, o.Dx]): Joint sample of target locations
    Pf (ndarray, [T, o.K, o.Dx, o.Dx]): Marginal target location covariances
      (Pf[t,k] is NOT the covariance from the posterior draw of x[t,k] because
       given x[t+1,k], this is just a function of the dynamics. Instead, it is
       the marginal covariance, which more clearly conveys are uncertainty
       about this portion of the joint sample.)
    ll (ndarray, [T,K]): log-likelihood of the joint sample, x
  """
  T = len(delta)
  x = np.zeros((T, o.K, o.Dx))
  Pf = np.zeros((T, o.K, o.Dx, o.Dx))
  ll = np.zeros((T, o.K))
  
  mu_k = list(zip(*mu))
  Sigma_k = list(zip(*Sigma))
  for k in range(o.K):
    x[:,k], _, _, _, Pf[:,k], ll[:,k] = mixtures.conjugate.sample_lds(
      mu_k[k], Sigma_k[k], o.Lambda[k], o.F, o.H, x0[k], P0[k])

  return x, Pf, ll

def CollectUsedParts(o, z, delta, pi, mu, Sigma):
  """ Remove truncated / non-associated parts.

  INPUT
    o (opts):
    z (ndarray, [o.N,]):
    delta (ndarray, [o.N,]):
    pi (ndarray, [o.K, o.maxBreaks,]):
    mu (ndarray, [o.K, o.maxBreaks, o.Dy_loc]):
    Sigma (ndarray, [o.K, o.maxBreaks, o.Dy_loc, o.Dy_loc])

  OUTPUT
    pi (ndarray, [ pi_1, pi_2, ..., pi_k ]
      pi_k (ndarray, [nParts_k]): mixture weights for target k
    mu (ndarray, [ mu_1, mu_2, ..., mu_k ]
      mu_k (ndarray, [nParts_k, o.Dy_loc])
    Sigma (ndarray, [ Sigma_1, Sigma_2, ..., Sigma_k ]
      Sigma_k (ndarray, [nParts_k, o.Dy_loc, o.Dy_loc])
  """
  _pi = [ [] for k in range(o.K) ]
  _mu = [ [] for k in range(o.K) ]
  _Sigma = [ [] for k in range(o.K) ]

  for k in range(o.K):
    delta_k = delta==k+1
    usedPartsIdx = np.unique( z[delta_k] ).astype(np.int)
    nUsedParts = len(usedPartsIdx)
    z[delta_k] = du.changem(z[delta_k], usedPartsIdx, range(nUsedParts))

    _pi[k] = pi[k, usedPartsIdx]
    _pi[k] = _pi[k] / np.sum(_pi[k])
    _mu[k] = mu[k, usedPartsIdx]
    _Sigma[k] = Sigma[k, usedPartsIdx]
  return z, _pi, _mu, _Sigma

def ShowSample(o, rgb, delta, z, mu, Sigma, x=None):
  if o.Dy_loc == 3: assert False, '3d visualization not ported yet.'
  if type(rgb) == str: rgb = du.imread(rgb)

  colors = du.diffcolors(o.K)
  for k in range(o.K):
    for p in range(len(mu[k])):
      deltak_zp = np.logical_and(delta==k+1, z==p)
      nAssociated = np.sum(deltak_zp)
      if nAssociated == 0: continue
      plt.plot(*du.stats.Gauss2DPoints(mu[k][p], Sigma[k][p]), c=colors[k])
      rgb = du.DrawOnImage(rgb, np.nonzero(du.asShape(deltak_zp, (o.nY, o.nX))),
        np.concatenate((colors[k], np.array([0.3,]))))
    if x is not None:
      plt.scatter(x[k,0], x[k,1], s=30, c=colors[k])

  plt.imshow(rgb)
  plt.ylim(0, o.nY)
  plt.xlim(0, o.nX)
  plt.gca().invert_yaxis()

def DrawSthSample(sampleDir, outDir, idx=-1):
  o = du.load('%s/opts' % sampleDir)
  imgs = du.load('%s/imgs' % sampleDir)
  samples = du.GetFilePaths(sampleDir, 'samples.*gz')
  sample = du.load(samples[idx])
  DrawSaveAllSamples(o, outDir, imgs, sample)

def DrawSaveAllSamples(o, outDir, imgs, sample):
  """ Draw all timesteps of sample (from build_sample).

  INPUT
    o (opts)
    outDir (str): location of output images
    imgs (list of strs): input images
    sample (argparse.Namespace): sample over time
  """
  T = len(imgs)
  outs = ['%s/img-%08d.jpg' % (outDir, t) for t in range(T)]
  pFunc = functools.partial(DrawSaveSample, o)
  args = list(zip(outs, imgs, sample.delta, sample.z, sample.mu,
    sample.Sigma))
  du.ParforD(pFunc, args)

def DrawSaveSample(o, saveName, rgb, delta, z, mu, Sigma, x=None, dpi=300):
  if o.Dy_loc == 3: assert False, '3d visualization not ported yet.'
  if type(rgb) == str: rgb = du.imread(rgb)

  fig, ax = plt.subplots()

  colors = du.diffcolors(o.K)
  for k in range(o.K):
    for p in range(len(mu[k])):
      deltak_zp = np.logical_and(delta==k+1, z==p)
      nAssociated = np.sum(deltak_zp)
      if nAssociated == 0: continue
      ax.plot(*du.stats.Gauss2DPoints(mu[k][p], Sigma[k][p]), c=colors[k])
      rgb = du.DrawOnImage(rgb, np.nonzero(du.asShape(deltak_zp, (o.nY, o.nX))),
        np.concatenate((colors[k], np.array([0.3,]))))
    if x is not None:
      ax.scatter(x[k,0], x[k,1], s=30, c=colors[k])

  ax.imshow(rgb)
  ax.set_ylim(0, o.nY)
  ax.set_xlim(0, o.nX)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.invert_yaxis()
  fig.savefig(saveName, dpi=dpi, bbox_inches='tight')
