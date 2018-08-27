import numpy as np, matplotlib.pyplot as plt
import du, du.stats
import IPython as ip, sys
import mixtures
from scipy.stats import multivariate_normal as mvn

def bg_test():
  dataPath = 'data/sample001'
  SigmaBA = 20*np.ones(3)
  SigmaBL = 5*np.ones(2)

  bg = du.imread('%s/bgRgb4x.png' % dataPath)
  bg = bg[:, 100:361]
  # bg = bg[:, 100:255]

  bgLab = du.rgb2labImg(bg)

  imgs = du.GetImgPaths('%s/rgb4x' % dataPath)
  img = du.imread(imgs[0])
  img = img[:, 100:361]
  # img = img[:, 100:255]
  imgLab = du.rgb2labImg(img)

  nY, nX, D = img.shape
  N = nY*nX
  
  xy = np.array(list(np.ndindex(nY, nX)))[:,::-1] 
  bL = xy.astype(np.float)
  bA = du.asShape(bgLab, (N, D)).astype(np.float)
  b = np.concatenate((bL, bA), axis=1)

  yL = xy.astype(np.float)
  yA = du.asShape(imgLab, (N, D)).astype(np.float)

  bgLL_A = du.stats.logmvnpdfdiag(yA, bA, SigmaBA)
  bgLL_L = du.stats.logmvnpdfdiag(xy, xy, SigmaBL)
  bgLL = bgLL_A + bgLL_L

  # set parameters and priors
  K = 2
  altLL = bgLL
  sideLL = -30 * np.ones((K, N))
  nSamples = 200
  maxBreaks = 5
  alpha = 0.1
  beta = np.array([98, 1, 1])

  v = 2000
  S = 20*(v - D - 1) * np.eye(2)

  Gamma = np.tile(1.0*np.eye(2), (K, 1, 1))
  zInitStrategy = 'zeros'
  x = np.array([
    [75, 75],
    [200, 200]])

  folder_name = '004'

  def run(idx):
    import mixtures
    return mixtures.fit_extents_model3( \
    yL, K, nSamples=nSamples, altLL=altLL, sideLL=sideLL,
    maxBreaks=maxBreaks, alpha=alpha, v=v, S=S,
    Gamma=Gamma, zInitStrategy=zInitStrategy, x=x,
    showProgress=False) # no Lambda

  z, delta, pi, Pi, mu, Sigma, ll = run(0)
  # res = du.ParforD(run, range(12))
  # z, delta, pi, Pi, mu, Sigma, ll = \
  #   res[np.argmax( [ r[-1][-1] for r in res ] ) ]

  colors = du.diffcolors(K)
  colors = np.concatenate(([[0, 0, 0]], colors))

  def show(idx, **kwargs):
    import matplotlib.pyplot as plt, numpy as np, du, du.stats
    if kwargs.get('figure') is not None:
      plt.figure(kwargs.get('figure').number)

    drawImg = img
    drawImgSolid = np.zeros((nY, nX, 3), dtype=np.uint8)
    deltaImg = du.asShape(delta[idx], (nY, nX))
    for k in range(K):
      dk = np.nonzero(deltaImg == k+1)
      drawImg = du.DrawOnImage(drawImg, dk,
        np.concatenate((colors[k+1], [0.25])))
      drawImgSolid = du.DrawOnImage(drawImgSolid, dk, colors[k+1])

    plt.subplot(121)
    plt.imshow(drawImg)
    for k in range(K):
      for p in range(maxBreaks):
        # don't draw components with no associations
        if np.sum(np.logical_and(z[idx] == p, delta[idx] == k+1)) < 1:
          continue

        if pi[idx, k, p] < 0.05:
          transparency = 0.1
          linestyle = '--'
        else:
          transparency = 0.3
          linestyle = '-'

        pts = du.stats.Gauss2DPoints(mu[idx,k,p], Sigma[idx,k,p])

        col = np.concatenate(( colors[k+1], [transparency,] ))
        plt.plot(*pts, c=col, linestyle=linestyle)
      plt.scatter(x[k, 0], x[k, 1], s=100, c=colors[k+1], marker='H')
    plt.xlim(0, nX)
    plt.ylim(0, nY)
    plt.gca().invert_yaxis()
    plt.title('Sample %05d' % idx)

    plt.subplot(122)
    plt.imshow(drawImgSolid)
    # plt.show()

  # save figures
  import os
  try: os.mkdir('extents/%s' % folder_name)
  except: None

  try: os.mkdir('extents/%s/samples' % folder_name)
  except: None

  def save(idx):
    import matplotlib.pyplot as plt
    f = plt.figure()
    show(idx, figure=f)
    plt.savefig('extents/%s/samples/%05d.png' % (folder_name, idx),
      bbox_inches='tight', dpi=270)
    plt.close(f.number)
  du.ParforD(save, range(nSamples))

  plt.plot(ll)
  plt.title('Sample Log-Likelihood')
  plt.savefig('extents/%s/ll.pdf' % folder_name, bbox_inches='tight')

  imgs = du.GetImgPaths('extents/%s/samples' % folder_name)
  du.rgbs2mp4(imgs, 'extents/%s/samples.mp4' % folder_name, crf=18, fps=10)

# dill.dump_session('extents/%s/session.pkl' % folder_name)



  # du.ViewPlots(range(nSamples), show)

  # fig, _ = du.ViewPlots(range(nSamples), show)
  # du.figure(num=fig.number, w=1600, h=1200)
  # plt.show()


def simple_test():
  N = 100
  D = 2
  y = np.random.rand(N, D)

  nSamples = 10
  K = 2
  altLL = -10*np.ones(N)
  sideLL = np.zeros((K, N))
  sideLL[0] = -10
  sideLL[1] = -20

  z, delta, pi, Pi, mu, Sigma, x, ll = \
    mixtures.fit_extents_model(y, K, nSamples=nSamples, altLL=altLL)

  cols = du.diffcolors(K+1)
  print(cols)

  plt.scatter(y[:,0], y[:,1], s=1, c=cols[delta[-1]])
  plt.show()

if __name__ == "__main__":
  bg_test()
  # simple_test()
