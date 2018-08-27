import numpy as np
from scipy.stats import multivariate_normal as mvn


def sample_lds(y, Sigma_y, Sigma_x, F, H, x0, P0):
  """ Sample from the joint posterior of a linear dynamical system.

  INPUT
    y (length-T list of ndarray): observations at each time
    Sigma_y (length-T list of ndarray): observation covariances at each time
    Sigma_x (ndarray, [Dx, Dx]): dispersion covariance
    F (ndarray, [Dx, Dx]): linear dynamics matrix
    H (ndarray, [Dy, Dx]): observation projection matrix
    x0 (ndarray, [Dx,]): initial target location
    P0 (ndarray, [Dx,Dx]): initial target location covariance

  OUTPUT   
    x (ndarray, [T, Dx]): target location samples at each time
    mu (ndarray, [T, Dx]): target joint posterior mean location parameter
    P (ndarray, [T, Dx, Dx]): target joint posterior cov location parameter
    xf (ndarray, [T, Dx]): target marginal posterior mean location parameter
    Pf (ndarray, [T, Dx, Dx]): target marginal posterior cov location parameter
    ll (ndarray, [T,]): log-likelihood of each sample
  """
  T = len(y)
  Dy, Dx = H.shape
  N = [ len(obs) for obs in y ]

  # forward filtering
  xf = np.zeros((T+1, Dx))
  Pf = np.zeros((T+1, Dx, Dx))
  xf[0] = x0; Pf[0] = P0
  for t in range(T):
    xf[t+1] = F.dot(xf[t])
    Pf[t+1] = F.dot(Pf[t]).dot(F.T) + Sigma_x
    for n in range(N[t]):
      xf[t+1], Pf[t+1] = filterUpdate(y[t][n], Sigma_y[t][n], H,
        xf[t+1], Pf[t+1])
  xf = xf[1:]
  Pf = Pf[1:]

  # backward sampling
  x = np.zeros((T, Dx))
  mu = np.zeros((T, Dx))
  P = np.zeros((T, Dx, Dx))
  ll = np.zeros(T)

  mu[-1] = xf[-1]
  P[-1] = Pf[-1]
  x[-1] = mvn.rvs(mu[-1], P[-1])
  ll[-1] = mvn.logpdf(x[-1], mu[-1], P[-1])
  for t in reversed(range(T-1)):
    mu[t], P[t] = normalUpdate(x[t+1], Sigma_x, xf[t], Pf[t], A=F)
    x[t] = mvn.rvs(mu[t], P[t])
    ll[t] = mvn.logpdf(x[t], mu[t], P[t])

  return x, mu, P, xf, Pf, ll

def smoothUpdate(F, mu_p, Sigma_p, mu_f, Sigma_f, mu_s, Sigma_s):
  """ RTS Smoothing.
  
  INPUT
    F: dynamics matrix
    mu_p, Sigma_p: prediction parameters from p(x_{t+1} | y_{1:t-1})
    mu_f, Sigma_f: filter parameters from p(x_t | y_{1:t})
    mu_s, Sigma_s: smooth parameters from p(x_{t+1} | y_{1:T})

  OUTPUT
    mu_s, Sigma_s: smooth parameters for p(x_t | y_{1:T})
  """
  J = Sigma_f.dot(F.T).dot(np.linalg.inv(Sigma_p))

  mu = mu_f + J.dot(mu_s - mu_p)
  Sigma = Sigma_f + J.dot(Sigma_s - Sigma_p).dot(J.T)
  return mu, Sigma

def filterUpdate(y, Sigma_y, H, mu_p, Sigma_p):
  """ Linear Gaussian Kalman Filtering.
  
  INPUT
    y: (N, Dy): N observations, y_t
    Sigma_y (Dy, Dy): (common) observation covariance
    H (Dy, Dx): observation projection matrix
    mu_p, Sigma_p: prediction parameters from p(x_{t+1} | y_{1:t-1})

  OUTPUT
    mu_f, Sigma_f: filter parameters from p(x_t | y_{1:t})
  """
  y = _process_observations(y)
  N, Dy = y.shape
  Dx = len(mu_p)
  I = np.eye(Dx)

  # H [Dy, Dx]
  # S [Dy, Dy]
  # Sigma [Dx, Dx]
  # K [Dx, Dy]
  # KH [Dx, Dx]

  mu = mu_p
  Sigma = Sigma_p
  for n in range(N):
    r = y[n] - H.dot(mu)
    S = H.dot(Sigma).dot(H.T) + Sigma_y # [Dy, Dy]
    K = Sigma.dot(H.T).dot(np.linalg.inv(S)) # [Dx,Dy]
    mu += K.dot(r)
    Sigma = (I - K.dot(H)).dot(Sigma) # [Dx, Dx]
  return mu, Sigma

def normalUpdate(y, Sigma, m0, V0, A=None):
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
  N, Dy = y.shape
  Dx = len(m0)
  assert Sigma.ndim == 2 and Sigma.shape[0] == Sigma.shape[1] == Dy, \
    'Sigma must be Dy x Dy covariance'
  assert V0.ndim == 2 and V0.shape[0] == V0.shape[1] == Dx, \
    'V0 must be Dx x Dx covariance'
  assert m0.ndim == 1 and len(m0) == Dx, 'm0 must be Dx-dimensional'
  if A is not None:
    assert A.shape[0] == Dy and A.shape[1] == Dx, 'A must be Dy x Dx'
  else:
    assert Dy == Dx, 'Dy != Dx and there is no projection'
    A = np.eye(Dy)

  # y : obs
  # Sigma : Sigma_y
  # m0 : x_{t-1}
  # V0 : Sigma_x
  # A  : Dy x Dx
  if N==0: return m0, V0

  yBar = np.mean(y, axis=0)
  V0i = np.linalg.inv(V0)
  SigmaI = np.linalg.inv(Sigma)
  A_SigmaI = A.T.dot(SigmaI)
  VN = np.linalg.inv(V0i + N * A_SigmaI.dot(A))
  mN = VN.dot( V0i.dot(m0) + N*A_SigmaI.dot(yBar) )

  return mN, VN

def _process_observations(Y):
  """ Enforce observations Y to be N x D. """
  Y = np.asarray(Y)
  if Y.ndim==1: Y = Y[np.newaxis,:]
  assert Y.ndim==2, 'Y must be N x D'
  return Y


def test1():
  import matplotlib.pyplot as plt, du, du.stats
  import IPython as ip, sys
  np.set_printoptions(precision=2, suppress=True)

  N = 10
  T = 20
  Dx, Dy = (4, 2)

  eye = np.eye(Dy)
  zer = np.zeros((Dy, Dy))
  F = np.block( [[eye, eye], [zer, eye]] )
  H = np.block( [eye, zer] )

  # Sigma_x = 5*np.eye(Dx)
  # Sigma_x = np.block( [[zer, zer], [zer, 5*eye]] )
  Sigma_x = np.block( [[0.01*eye, zer], [zer, 5*eye]] )
  Sigma_y = 5*np.eye(Dy)
  
  # sample data
  x = np.zeros((T, Dx))
  y = np.zeros((T, N, Dy))

  for t in range(1,T): x[t] = mvn.rvs(F.dot(x[t-1]), Sigma_x)
  for t in range(T): y[t] = mvn.rvs(H.dot(x[t]), Sigma_y, size=N)

  # filter (starting with true x)
  mu_f = np.zeros((T+1, Dx))
  Sigma_f = np.zeros((T+1, Dx, Dx))

  mu_p = np.zeros((T, Dx))
  Sigma_p = np.zeros((T, Dx, Dx))

  mu_s = np.zeros((T, Dx))
  Sigma_s = np.zeros((T, Dx, Dx))

  # initial values
  Sigma_f[0] = 1e2*np.eye(Dx)
  
  # filter
  for t in range(T):
    # predict
    mu_p[t] = F.dot(mu_f[t])
    Sigma_p[t] = F.dot(Sigma_f[t]).dot(F.T) + Sigma_x
    mu_f[t+1], Sigma_f[t+1] = filterUpdate(y[t], Sigma_y, H,
      mu_p[t], Sigma_p[t])
  mu_f = mu_f[1:]
  Sigma_f = Sigma_f[1:]

  # smooth
  mu_s[-1] = mu_f[-1]
  Sigma_s[-1] = Sigma_f[-1]
  for t in reversed(range(T-1)):
    mu_s[t], Sigma_s[t] = smoothUpdate(F, mu_p[t+1], Sigma_p[t+1],
      mu_f[t], Sigma_f[t], mu_s[t+1], Sigma_s[t+1])

  # forward filtering, backward sampling
  mu_post = np.zeros((T, Dx))
  Sigma_post = np.zeros((T, Dx, Dx))
  mu_post[-1] = mu_f[-1]
  Sigma_post[-1] = Sigma_f[-1]

  xs = np.zeros((T, Dx))
  xs[-1] = mvn.rvs(mu_post[-1], Sigma_post[-1])
  for t in reversed(range(T-1)):
    # xs[t] | xs[t+1] propto
    #   N(xs[t] | mu_f, Sigma_f) (prior)
    #   N(xs[t+1] | F xs[t], Sigma_x) (likelihood)
    mu_post[t], Sigma_post[t] = normalUpdate(xs[t+1],
      Sigma_x, mu_f[t], Sigma_f[t], A=F)
    xs[t] = mvn.rvs(mu_post[t], Sigma_post[t])

  # visualize
  xlim = [np.minimum(np.min(x[:,0]), np.min(y[:,:,0]))-5,
          np.maximum(np.max(x[:,0]), np.max(y[:,:,0]))+5]
  ylim = [np.minimum(np.min(x[:,1]), np.min(y[:,:,1]))-5,
          np.maximum(np.max(x[:,1]), np.max(y[:,:,1]))+5]
  trueColor = [0, 0, 1, 0.5]
  obsColor = [0, 0, 0, 0.5]
  filterColor = [0, 1, 0, 0.5]
  smoothColor = [1, 0, 0, 0.5]
  posteriorColor = [1, 0, 1, 0.5]

  def show(t):
    plt.scatter(x[t,0], x[t,1], c=trueColor, s=50, label='trueX')
    plt.scatter(y[t,:,0], y[t,:,1], c=obsColor, s=10, label='obs')
    plt.plot(*du.stats.Gauss2DPoints(H.dot(mu_f[t]), H.dot(Sigma_f[t]).dot(H.T)),
      c=filterColor, linestyle='--', label='filter')
    plt.plot(*du.stats.Gauss2DPoints(H.dot(mu_s[t]), H.dot(Sigma_s[t]).dot(H.T)),
      c=posteriorColor, linestyle='-.', label='posterior')

    plt.scatter(xs[t,0], xs[t,1], c=posteriorColor, s=30, label='posteriorX')
    plt.plot(*du.stats.Gauss2DPoints(H.dot(mu_post[t]), H.dot(Sigma_post[t]).dot(H.T)),
      c=smoothColor, linestyle='--', label='snooth')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.title('Time %05d' % t)

  du.ViewPlots(range(T), show)
  plt.show()

def test2():
  import matplotlib.pyplot as plt, du, du.stats
  import IPython as ip, sys
  np.set_printoptions(precision=2, suppress=True)

  N = 10
  T = 10
  Dx, Dy = (4, 2)

  eye = np.eye(Dy)
  zer = np.zeros((Dy, Dy))
  F = np.block( [[eye, eye], [zer, eye]] )
  H = np.block( [eye, zer] )

  Sigma_x = np.block( [[0.01*eye, zer], [zer, 5*eye]] )
  Sigma_y = 5*np.eye(Dy)
  
  # sample data
  x = np.zeros((T, Dx))
  y = np.zeros((T, N, Dy))

  for t in range(1,T): x[t] = mvn.rvs(F.dot(x[t-1]), Sigma_x)
  for t in range(T): y[t] = mvn.rvs(H.dot(x[t]), Sigma_y, size=N)

  # Format args
  y_args = [ yt for yt in y ]
  Sigma_y_args = [ np.tile(Sigma_y[np.newaxis], [N, 1, 1]) for yt in y]
  # Sigma_y_args = np.tile(Sigma_y[np.newaxis, np.newaxis, :, :], [T, N, 1, 1])

  # test removing all observations for a given time
  # y_args[3] = []
  # Sigma_y_args[3] = []

  # test removing some observations for a given time
  # y_args[3] = y_args[3][:5]
  # Sigma_y_args[3] = Sigma_y_args[3][:5]

  # test removing all observations for all times
  y_args = [ [] for yt in y ]
  Sigma_y_args = [ [] for yt in y ]

  x0 = np.zeros(Dx)
  P0 = 1e5 * np.eye(Dx)
  xs, mu, P, xf, Pf, ll = sample_lds(y_args, Sigma_y_args, Sigma_x, F, H, x0, P0)

  # visualize
  xlim = [np.minimum(np.min(x[:,0]), np.min(y[:,:,0]))-5,
          np.maximum(np.max(x[:,0]), np.max(y[:,:,0]))+5]
  ylim = [np.minimum(np.min(x[:,1]), np.min(y[:,:,1]))-5,
          np.maximum(np.max(x[:,1]), np.max(y[:,:,1]))+5]
  trueColor = [0, 0, 1, 0.5]
  obsColor = [0, 0, 0, 0.5]
  sampleColor = [1, 0, 1, 0.5]

  def show(t):
    plt.scatter(x[t,0], x[t,1], c=trueColor, s=50, label='trueX')
    plt.scatter(y[t,:,0], y[t,:,1], c=obsColor, s=10, label='obs')

    plt.plot(*du.stats.Gauss2DPoints(H.dot(mu[t]), H.dot(P[t]).dot(H.T)),
      c=sampleColor, linestyle='--', label='joint sample dist')
    plt.scatter(xs[t,0], xs[t,1], c=sampleColor, s=30, label='joint sample x')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.title('Time %05d' % t)

  du.ViewPlots(range(T), show)
  plt.show()

  ip.embed()
  # plt.plot(ll)
  # plt.show()

if __name__ == "__main__":
  test2()
