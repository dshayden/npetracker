import numpy as np, matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn, invwishart as iw
import mixtures
import du, du.stats
import IPython as ip
np.set_printoptions(precision=2, suppress=True)

def test1():
  D = 2
  K = 3
  N = 10

  mu = np.array([
    [-0.25, 0.25],
    [ 0.25, 0.25],
    [   0, -0.25]])
  Sigma = np.stack( [iw.rvs(3, np.eye(D)) for k in range(K)] )
  Sigma = Sigma * 0.05
  theta = [ (mu[k], Sigma[k]) for k in range(K) ]

  pi = np.array([100, 50, 50])
  pi = pi / np.sum(pi)

  nPts = 1000
  vals = np.linspace(-5, 5, nPts)
  x, y = np.meshgrid(vals, vals)
  xy = np.stack( (x.flatten(), y.flatten()) ).T

  ll = mixtures.gmm_logpdf_marginalized(xy, theta, pi)
  like = np.exp(ll)

  for k in range(K):
    plt.plot(*du.stats.Gauss2DPoints(*theta[k]), '--')

  plt.contourf(x, y, du.asShape(like, (nPts, nPts)))
  plt.xlim(-2, 2)
  plt.ylim(-2, 2)
  plt.show()

def test2():
  # parameters
  K = 3
  D = 2
  N = 1000

  # set true (shared) priors
  alpha = np.ones(K)
  dirmul = mixtures.conjugate.dirmul(alpha)

  m0 = np.zeros(D)
  kappa = 0.01
  v = 5
  S = np.eye(D)
  phi = mixtures.conjugate.nniw(m0, kappa, S, v)

  # simulate data
  pi_true = dirmul.sample()
  theta_true = [ phi.sample() for k in range(K) ]

  y = np.concatenate( [mvn.rvs(*theta_true[k], size=int(pi_true[k]*N))
    for k in range(K)] )
  z_true = np.concatenate( [k*np.ones(int(pi_true[k]*N)) for k in range(K) ] )
  z_true = z_true.astype(np.int)
  N = y.shape[0]

  # run inference with correct priors
  nSamples = 50
  res = mixtures.gmm_fit_blocked_gibbs(y, dirmul, [phi for k in range(K)],
    nSamples=nSamples)
  z, pi, theta, ll = res

  # plot results
  colors = du.diffcolors(K)

  plt.figure()
  plt.plot(ll)
  plt.title('Sample Log-Likelihood')

  nContourPts = 1000
  xvals = np.linspace(np.min(y[:,0]), np.max(y[:,0]), nContourPts)
  yvals = np.linspace(np.min(y[:,1]), np.max(y[:,1]), nContourPts)
  xx, yy = np.meshgrid(xvals, yvals)
  xy = np.stack( (xx.flatten(), yy.flatten()) ).T

  plt.figure()
  plt.subplot(211)
  # cf = np.exp(mixtures.gmm_logpdf_marginalized(xy, theta_true, pi_true))
  # plt.contourf(xx, yy, du.asShape(cf, (nContourPts, nContourPts)),
  #   alpha=0.25)
  for k in range(K):
    plt.plot(*du.stats.Gauss2DPoints(*theta_true[k]), '--', c=colors[k])
  plt.scatter(y[:,0], y[:,1], s=1, c=colors[z_true])

  cf = np.exp(mixtures.gmm_logpdf_marginalized(xy, theta_true, pi_true))
  plt.contour(xx, yy, du.asShape(cf, (nContourPts, nContourPts)),
    alpha=1)

  plt.title('True Clusters and Labels')

  plt.subplot(212)
  # cf_sample = np.exp(mixtures.gmm_logpdf_marginalized(xy, theta[-1], pi[-1]))
  # plt.contourf(xx, yy, du.asShape(cf_sample, (nContourPts, nContourPts)),
  #   alpha=0.25)
  for k in range(K):
    plt.plot(*du.stats.Gauss2DPoints(*theta[-1][k]), '--', c=colors[k])
  plt.scatter(y[:,0], y[:,1], s=1, c=colors[z[-1]])

  cf_sample = np.exp(mixtures.gmm_logpdf_marginalized(xy, theta[-1], pi[-1]))
  plt.contour(xx, yy, du.asShape(cf_sample, (nContourPts, nContourPts)),
    alpha=1)

  plt.title('Sampled Clusters and Labels')
  
  plt.show()
  

if __name__ == "__main__":
  test2()
