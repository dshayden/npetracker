import numpy as np, matplotlib.pyplot as plt
import NPETracker as npe
import os
import du, du.stats
import IPython as ip, sys
import dill
np.set_printoptions(precision=2, suppress=True)

dataPath = 'data/sample001'
imgs = du.GetImgPaths('%s/rgb4x' % dataPath)
deps = du.GetImgPaths('%s/depth2color4x' % dataPath)
fg = du.imread('%s/fgMask4x.png' % dataPath)
bgImg = du.imread('%s/bgRgb4x.png' % dataPath)
bgDep = du.imread('%s/bgDep.png' % dataPath)
nY, nX = bgImg.shape[:2]
K = 2
Dy_loc = 2
T = len(imgs)

nSamples = 1000
outdirSamples = 'samples/00007'
outdirImages = 'samples/00007/images'
prevX = np.tile( [[0, 0, 0, 0], [480, 270, 0, 0]], [T, 1, 1] )

v = 5000
S = 25 * (v - Dy_loc - 1) * np.eye(Dy_loc)
# Gamma = np.tile(100*np.eye(Dy_loc), (K, 1, 1))
Gamma = np.tile(np.eye(Dy_loc), (K, 1, 1))
nPartSamples = 5
maxBreaks = 5
o = npe.build_opts(nY, nX, K, fgMask=fg, nPartSamples=nPartSamples,
  v=v, S=S, Gamma=Gamma, maxBreaks=maxBreaks)

# set background model
bgMu, bg_mask = npe.LoadObs(o, bgImg, bgDep)
Sigma_B = 50 * np.ones((o.N, o.Dy))
bg = npe.BGModel(bgMu, Sigma_B, bg_mask)

# set target appearance models as highly vague
pi_A = np.array([1,])
mu_A = 128*np.ones(3)
Sigma_A = 1e8*np.eye(3)
app = [ npe.AppModel(pi_A, mu_A[np.newaxis], Sigma_A[np.newaxis])
  for k in range(o.K) ]

def joint_sample(prevX):
  """ Construct joint sample p(x, z, delta, mu, Sigma, pi, Pi | Y).

  INPUT
    prevX (ndarray, [T, o.K, o.Dx]): previous samples of x

  OUTPUT
    z, delta, pi, Pi, mu, Sigma, ll_t, x, Pf, ll_x
  """
  args = [ (o, prevX[t], bg, app, imgs[t], deps[t]) for t in range(T) ]
  res = du.Parfor(npe.SampleParts_t, args)
  z, delta, pi, Pi, mu, Sigma, ll_t = zip(*res)
  x, Pf, ll_x = npe.SampleLocs(o, delta, z, mu, Sigma, o.x0, o.P0)
  ll = np.sum(ll_t) + np.sum(ll_x)
  s = npe.build_sample(o, x, Pf, mu, Sigma, Pi, pi, ll, delta=delta,
    z=z)
  return s

# T = 5
try: os.mkdir(outdirSamples)
except: None
assert os.path.exists(outdirSamples), "Can't save output, failing."
try: os.mkdir(outdirImages)
except: None

du.save('%s/opts' % outdirSamples, o)
du.save('%s/imgs' % outdirSamples, imgs)
last_sample = None
du.tic()
for s in range(nSamples):
  print('Sample %05d, Total Mins: %.2f' % (s, du.toc() / 60.))
  last_sample = joint_sample(prevX)
  du.save('%s/sample-%05d' % (outdirSamples, s), last_sample)
  prevX = last_sample.x

npe.DrawSthSample(outdirSamples, outdirImages)


# def show(t):
#   ls = last_sample
#   npe.ShowSample(o, imgs[t], ls.delta[t], ls.z[t], ls.mu[t], ls.Sigma[t], ls.x[t])
# du.ViewPlots(range(T), show)
# plt.show()
