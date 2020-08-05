import copy
import logging
import sys
import time

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse as scisp
import scipy.ndimage.filters as filters
import skimage.io

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import core.sparse as sp
import optim as optim

from torchlib.modules import LinearChain
from torchlib.modules import SkipAutoencoder
from torchlib.image import crop_like

log = logging.getLogger(__name__)

import scipy.sparse as ssp
def matlab_dump(sp, M, N):
  sp2 = ssp.csr_matrix(
      (sp.val.cpu().data.numpy(),
      sp.col_idx.cpu().data.numpy(),
      sp.csr_row_idx.cpu().data.numpy()),
      shape=(M, N)).tocoo()
  return sp2

class MattingCNN(nn.Module):
  def __init__(self, cg_steps=200):
    super(MattingCNN, self).__init__()

    self.cg_steps = cg_steps

    self.net = SkipAutoencoder(4, 4, width=16, depth=5, batchnorm=True, grow_width=False)
    self.weight_normalizer = th.nn.Softmax2d()
    self.system = MattingSystem()
    self.solver = MattingSolver(steps=cg_steps)

    self.reset_parameters()

  def reset_parameters(self):
    pass
    # self.net.prediction.bias.data[0] = 1
    # self.net.prediction.bias.data[1] = 1
    # self.net.prediction.bias.data[2] = 1
    # self.net.prediction.bias.data[3] = 1
    # self.net.prediction.bias.data[4] = 1
    # self.net.prediction.weight.data.normal_(0, 0.001)

  def forward(self, sample):
    assert sample['image'].shape[0] == 1  # NOTE: we do not handle batches at this point
    h = sample['image'].shape[2]
    w = sample['image'].shape[3]
    N = h*w
    # force non-negative weights
    eps = 1e-8
    weights = self.net(th.cat([sample['image'], sample['trimap']], 1))
    weights = self.weight_normalizer(weights)
    self.predicted_weights = weights
    weights = weights.view(4, h*w)

    CM_weights  = weights[0, :]
    LOC_weights = weights[1, :]
    IU_weights  = weights[2, :]
    KU_weights  = weights[3, :]
    # lmbda       = weights[4, :]

    lmbda = Variable(th.from_numpy(np.array([100.0], dtype=np.float32)).cuda(),
                     requires_grad=False)

    single_sample = {}
    for k in sample.keys():
      if "Tensor" not in type(sample[k]).__name__:
        single_sample[k] = sample[k][0, ...]

    A, b = self.system(
        single_sample, CM_weights, LOC_weights,
        IU_weights, KU_weights, lmbda, N)
    matte = self.solver(A, b)
    residual = self.solver.err
    matte = matte.view(1, 1, h, w)
    matte = th.clamp(matte, 0, 1)
    log.info("CG residual: {:.1f} in {} steps".format(residual, self.solver.stop_step))
    if residual < 0:
      import ipdb; ipdb.set_trace()

    return matte


class MattingSolver(nn.Module):
  def __init__(self, steps=30, verbose=False):
    self.steps = steps
    self.verbose = verbose
    super(MattingSolver, self).__init__()

  def forward(self, A, b):
    start = time.time()
    x0 = Variable(th.zeros(b.shape[0]).cuda(), requires_grad=False)
    x_opt, err, stop_step = optim.sparse_cg(A, b, x0, steps=self.steps, verbose=self.verbose)
    end = time.time()
    if self.verbose:
      log.debug("solve system {:.2f}s".format((end-start)))
    self.err = err
    self.stop_step = stop_step
    return x_opt


class MattingSystem(nn.Module):
  """docstring for MattingSystem"""
  def __init__(self):
    super(MattingSystem, self).__init__()
    
  def forward(self, sample, CM_weights, LOC_weights, IU_weights, KU_weights, lmbda, N):
    start = time.time()
    Lcm = self._color_mixture(N, sample, CM_weights)
    Lmat = self._matting_laplacian(N, sample, LOC_weights)
    Lcs = self._intra_unknowns(N, sample, IU_weights)

    kToUconf = sample['kToUconf']
    known = sample['known']
    kToU = sample['kToU']

    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())
    linear_csr_row_idx = Variable(th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda())

    KU = sp.Sparse(linear_csr_row_idx, linear_idx, KU_weights.mul(kToUconf), th.Size((N,N)))
    known = sp.Sparse(linear_csr_row_idx, linear_idx, lmbda.mul(known), th.Size((N,N)))

    # A = sp.spadd(Lmat, sp.spadd(KU, known))
    # A = sp.spadd(Lcm, sp.spadd(sp.spadd(KU, known), Lcs))
    A = sp.spadd(Lcm, sp.spadd(Lmat, sp.spadd(sp.spadd(KU, known), Lcs)))
    b = sp.spmv(sp.spadd(KU, known), kToU)

    end = time.time()
    log.debug("prepare system {:.2f}s/im".format((end-start)))

    return A, b

  def _color_mixture(self, N, sample, CM_weights):
    # CM
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())
    linear_csr_row_idx = Variable(th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda())

    Wcm = sp.from_coo(sample["Wcm_row"], sample["Wcm_col"].view(-1),
                      sample["Wcm_data"], th.Size((N, N)))

    diag = sp.Sparse(linear_csr_row_idx, linear_idx, CM_weights, th.Size((N, N)))
    Wcm = sp.spmm(diag, Wcm)
    ones = Variable(th.ones(N).cuda())
    row_sum = sp.spmv(Wcm, ones)
    Wcm.mul_(-1.0)
    Lcm = sp.spadd(sp.from_coo(linear_idx, linear_idx, row_sum.data, th.Size((N, N))), Wcm)
    Lcmt = sp.transpose(Lcm)
    Lcm = sp.spmm(Lcmt, Lcm)
    return Lcm

  def _matting_laplacian(self, N, sample, LOC_weights):
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())

    w = sample['image'].shape[-1]
    h = sample['image'].shape[-2]

    # Matting Laplacian
    inInd = sample["LOC_inInd"]
    weights = LOC_weights[inInd.long().view(-1)]
    flows = sample['LOC_flows']
    flow_sz = flows.shape[0]
    tiled_weights = weights.view(1, 1, -1).repeat(flow_sz, flow_sz, 1)
    flows = flows.mul(tiled_weights)

    neighInds = th.cat(
        [inInd-1-w, inInd-1, inInd-1+w, inInd-w, inInd, inInd+w, inInd+1-w, inInd+1, inInd+1+w], 1)

    for i in range(9):
      iRows = neighInds[:, i:i+1].clone().repeat(1, 9)
      iFlows = flows[:, i, :].contiguous().permute(1, 0).clone()
      iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), iFlows.view(-1), th.Size((N, N)))
      if i == 0:
        Wmat = iWmat
      else:
        Wmat = sp.spadd(Wmat, iWmat)

    Wmatt = sp.transpose(Wmat)
    Wmat = sp.spadd(Wmat, Wmatt)
    Wmat.mul_(0.5)
    ones = Variable(th.ones(N).cuda())
    row_sum = sp.spmv(Wmat, ones)
    Wmat.mul_(-1.0)
    diag = sp.from_coo(linear_idx, linear_idx, row_sum, th.Size((N, N)))
    Lmat = sp.spadd(diag, Wmat)
    return Lmat

  def _matting_laplacian_verbose(self, N, sample, LOC_weights):
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())

    w = sample['image'].shape[-1]
    h = sample['image'].shape[-2]

    # Matting Laplacian
    inInd = sample["LOC_inInd"]
    scipy.io.savemat("indices.mat", {
      "pyLOC_inInd": inInd.cpu().data.numpy(),
      })
    weights = LOC_weights[inInd.long().view(-1)]
    flows1 = sample['LOC_flows1']
    flows2 = sample['LOC_flows2']
    flows3 = sample['LOC_flows3']
    flows4 = sample['LOC_flows4']
    flows5 = sample['LOC_flows5']
    flows6 = sample['LOC_flows6']
    flows7 = sample['LOC_flows7']
    flows8 = sample['LOC_flows8']
    flows9 = sample['LOC_flows9']
    flow_sz = flows1.shape[1]
    tiled_weights = weights.view(-1, 1).repeat(1, flow_sz)

    flows1 = flows1.mul(tiled_weights)
    flows2 = flows2.mul(tiled_weights)
    flows3 = flows3.mul(tiled_weights)
    flows4 = flows4.mul(tiled_weights)
    flows5 = flows5.mul(tiled_weights)
    flows6 = flows6.mul(tiled_weights)
    flows7 = flows7.mul(tiled_weights)
    flows8 = flows8.mul(tiled_weights)
    flows9 = flows9.mul(tiled_weights)

    neighInds = th.cat(
        [inInd-1-w, inInd-1, inInd-1+w, inInd-w, inInd, inInd+w, inInd+1-w, inInd+1, inInd+1+w], 1)
    scipy.io.savemat("neigh_indices.mat", {
      "pyLOC_neighInd": neighInds.cpu().data.numpy(),
      })

    iRows = neighInds[:, 0:1].clone().repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows1.view(-1), th.Size((N, N)))
    Wmat = iWmat
    l = matlab_dump(Wmat, N, N)
    scipy.io.savemat("Wmat1.mat", {
      "py_wmat1_irows": iRows.cpu().data.numpy(),
      "py_wmat1_row": l.row,
      "py_wmat1_col": l.col,
      "py_wmat1_data": l.data,
      })

    iRows = neighInds[:, 1:2].clone().repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows2.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)
    l = matlab_dump(iWmat, N, N)
    scipy.io.savemat("Wmat2.mat", {
      "py_wmat2_irows": iRows.cpu().data.numpy(),
      "py_wmat2_row": l.row,
      "py_wmat2_col": l.col,
      "py_wmat2_data": l.data,
      })

    iRows = neighInds[:, 2].clone().view(-1, 1).repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows3.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)

    iRows = neighInds[:, 3].clone().view(-1, 1).repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows4.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)

    iRows = neighInds[:, 4].clone().view(-1, 1).repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows5.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)

    iRows = neighInds[:, 5].clone().view(-1, 1).repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows6.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)

    iRows = neighInds[:, 6].clone().view(-1, 1).repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows7.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)

    iRows = neighInds[:, 7].clone().view(-1, 1).repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows8.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)

    iRows = neighInds[:, 8].clone().view(-1, 1).repeat(1, 9)
    iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), flows9.view(-1), th.Size((N, N)))
    Wmat = sp.spadd(Wmat, iWmat)

    l = matlab_dump(Wmat, N, N)
    scipy.io.savemat("dump.mat", {
      "py_row": l.row,
      "py_col": l.col,
      "py_data": l.data,
      })

    Wmat_transp = sp.transpose(Wmat)
    Wmat = sp.spadd(Wmat, Wmat_transp)
    Wmat.val *= (0.5)
    ones = Variable(th.ones(N).cuda())
    row_sum = sp.spmv(Wmat, ones)
    Wmat.mul_(-1.0)
    diag = sp.from_coo(linear_idx, linear_idx, row_sum, th.Size((N, N)))
    Lmat = sp.spadd(diag, Wmat)

    # import ipdb; ipdb.set_trace()

    return Lmat

  def _intra_unknowns(self, N, sample, IU_weights):
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())

    weights = IU_weights[sample["IU_inInd"].long().view(-1)]
    nweights = weights.numel()
    flows = sample['IU_flows']
    flow_sz = flows.shape[1]
    flows = flows.mul(weights.view(-1, 1).repeat(1, flow_sz))
    neighInd = sample["IU_neighInd"].contiguous()
    inInd = sample["IU_inInd"].clone()
    inInd = inInd.repeat(1, neighInd.shape[1])
    Wcs = sp.from_coo(inInd.view(-1), neighInd.view(-1), flows.data.view(-1), th.Size((N, N)))
    Wcst = sp.transpose(Wcs)
    Wcs = sp.spadd(Wcs, Wcst)
    Wcs.mul_(0.5)
    ones = Variable(th.ones(N).cuda())
    row_sum = sp.spmv(Wcs, ones)
    Wcs.mul_(-1)
    diag = sp.from_coo(linear_idx, linear_idx, row_sum.data, th.Size((N, N)))
    Lcs = sp.spadd(diag, Wcs)
    return Lcs


class CharbonnierLoss(nn.Module):
  def __init__(self, epsilon=1e-6):
    super(CharbonnierLoss, self).__init__()
    self.epsilon = epsilon

  def forward(self, output, target): 
    diff  = th.sqrt((th.pow(output-target, 2) + self.epsilon*self.epsilon))
    return diff.mean()


class AlphaSparsity(nn.Module):
  def __init__(self):
    super(AlphaSparsity, self).__init__()

  def forward(self, alpha):
    s = th.pow(alpha, 2) + th.pow(1-alpha, 2)
    return 1.0/s - 1.0


class AlphaGradientNorm(nn.Module):
  def __init__(self, blur_std=1, truncation=3):
    super(AlphaGradientNorm, self).__init__()
    self.std = blur_std
    self.truncation = truncation
    n = int(np.ceil(self.truncation*self.std))
    self.n = n

    self.gaussian_x = nn.Conv2d(1, 1, (1, 2*n+1), bias=False)
    self.gaussian_y = nn.Conv2d(1, 1, (2*n+1, 1), bias=False)
    self.dx = nn.Conv2d(1, 1, (1, 2), bias=False)
    self.dy = nn.Conv2d(1, 1, (2, 1), bias=False)

    kernel = np.zeros((2*n+1, 1))
    kernel[n, 0] = 1.0
    kernel = filters.gaussian_filter1d(kernel, self.std, truncate=self.truncation).astype(np.float32)
    self.gaussian_x.weight.data[0, 0, 0, :] = th.from_numpy(kernel)
    self.gaussian_y.weight.data[0, 0, :, 0] = th.from_numpy(kernel)
    self.dx.weight.data[0, 0, 0, :] = th.from_numpy(np.array([-1, 1], dtype=np.float32))
    self.dy.weight.data[0, 0, :, 0] = th.from_numpy(np.array([-1, 1], dtype=np.float32))

  def forward(self, alpha):
    alpha = self.gaussian_x(alpha)
    alpha = self.gaussian_y(alpha)
    dx = self.dx(alpha[:, :, :-1, :])
    dy = self.dy(alpha[:, :, :, :-1])
    return th.abs(dx) + th.abs(dy)


class AlphaLoss(nn.Module):
  def __init__(self, sparsity_weight=1.0, gradient_weight=1.0, epsilon=1e-6, blur_std=1, truncation=3):
    super(AlphaLoss, self).__init__()

    self.epsilon = epsilon
    self.sparsity_weight = sparsity_weight
    self.gradient_weight = gradient_weight
    self.sparsity = AlphaSparsity()
    self.gradient = AlphaGradientNorm()

  def forward(self, output, target): 
    diff  = th.sqrt((th.pow(output-target, 2) + self.epsilon*self.epsilon))
    sparsity = self.sparsity_weight*self.sparsity(target)
    gradient_norm = self.gradient_weight*self.gradient(target)

    # Crop to match gradient term
    diff = crop_like(diff, gradient_norm)
    sparsity = crop_like(sparsity, gradient_norm)
    diff = diff[:, :, :-1, :-1]
    sparsity = sparsity[:, :, :-1, :-1]

    diff *= (1 + sparsity + gradient_norm)
    return diff.mean()


def get(params):
  params = copy.deepcopy(params)  # do not touch the original
  model_name = params.pop("model", None)
  return getattr(sys.modules[__name__], model_name)(**params)
