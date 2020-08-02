import logging

import torch as th

import matting.sparse as sp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def cg(A, b, x0, steps=1, thresh=1e-5, verbose=False):
  r = b - A.matmul(x0)
  p = r.clone()
  x = x0.clone()
  res_old = r.dot(r)
  err = -1

  for k in range(steps):
    Ap = A.matmul(p)
    alpha = res_old / p.dot(Ap)
    x = x +  alpha*p
    r = r - alpha*Ap
    res_new = r.dot(r)
    err = th.sqrt(res_new).data[0]
    if (err < thresh):
      if verbose:
        log.info("CG converged with residual {}.".format(err))
      break
    if verbose:
      log.info("CG step {} / {}, residual = {:g}".format(k+1, steps, err))
    p = r + res_new/res_old*p
    res_old = res_new
  return x, err


def sparse_cg(A, b, x0, steps=1, thresh=1e-4, verbose=False):
  r = b - sp.spmv(A, x0)
  p = r.clone()
  x = x0.clone()
  res_old = r.dot(r)
  err = -1

  for k in range(steps):
    Ap = sp.spmv(A, p)
    alpha = res_old / p.dot(Ap)
    x = x +  alpha*p
    r = r - alpha*Ap
    res_new = r.dot(r)
    err = th.sqrt(res_new).data[0]
    if (err < thresh):
      if verbose:
        log.info("CG converged with residual {}.".format(err))
      break
    if verbose:
      log.info("CG step {} / {}, residual = {:g}".format(k+1, steps, err))
    p = r + res_new/res_old*p
    res_old = res_new
  return x, err, k+1
