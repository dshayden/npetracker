import numpy as np

def catrnd(p):
  """ Sample categorical random variables.

  Args:
    p (ndarray): NxD array, N separate trials each with D outcomes. Assumes that
                 the columns of p all sum to 1.
  """
  cs = np.cumsum(p, axis=1)
  rv = np.random.rand(p.shape[0])[:,np.newaxis]
  return np.argmax(rv <= cs, axis=1)

def process_observations(Y):
  """ Ensure observations Y are NxD. """
  Y = np.asarray(Y)
  if Y.ndim==1: Y = Y[np.newaxis,:]
  return Y
