import numpy as np
import torch as th
import matting.functions.sparse as spfuncs

from torch.autograd import Variable

import scipy.sparse as scp


class Sparse(object):
  """"""
  def __init__(self, csr_row_idx, col_idx, val, size):
    if csr_row_idx.numel() != size[0]+1:
      raise ValueError("CSR row should have rows+1 elements")
    if col_idx.numel() != val.numel():
      raise ValueError("Col and Val should have the same number of elements.")
    if col_idx.numel() > size[0]*size[1]:
      raise ValueError("NNZ should be less than rows*cols.")
    self.csr_row_idx = csr_row_idx
    self.col_idx = col_idx
    self.val = val
    self.size = size
    self.storage = "csr"

  def make_variable(self, requires_grad=False):
    self.csr_row_idx = Variable(self.csr_row_idx)
    self.col_idx = Variable(self.col_idx)
    self.val = Variable(self.val, requires_grad=requires_grad)

  @property
  def nnz(self):
    return self.val.numel()

  def mul_(self, s):
    self.val = self.val.mul(s)
    # self.val.mul_(s)

  def to_dense(self):
    vals = self.val.data.cpu()
    ptr = self.csr_row_idx.data.cpu()
    cols = self.col_idx.data.cpu()
    size = self.size
    mat = scp.csr_matrix((vals, cols, ptr), shape=(size[0], size[1]))
    return mat.todense()

  def __str__(self):
    s = "Sparse matrix {}\n".format(self.size)
    s += "  csr_row {}\n".format(self.csr_row_idx)
    s += "  col {}\n".format(self.col_idx)
    s += "  val {}\n".format(self.val)
    return s


def from_coo(row_idx, col_idx, val, size):
  """Construct a sparse matrix from THTensors describing a COO format."""
  if row_idx.numel() != col_idx.numel():
    raise ValueError("Row and Col should have the same number of elements.")
  if row_idx.numel() != val.numel():
    raise ValueError("Row and Val should have the same number of elements.")
  if row_idx.numel() > size[0]*size[1]:
    raise ValueError("NNZ should be less than rows*cols.")
  csr_row_idx, csr_col_idx, csr_val = spfuncs.Coo2Csr.apply(row_idx, col_idx, val, size)
  return Sparse(csr_row_idx, csr_col_idx, csr_val, size)


def size_to_variable(size):
  asize = np.array(list(size)).astype(np.int32)
  asize = Variable(th.from_numpy(asize))
  return asize


def transpose(A):
  # asize = size_to_variable(A.size)
  csc_row_idx, csc_col_idx, csc_val = spfuncs.Transpose.apply(A.csr_row_idx, A.col_idx, A.val, A.size)
  return Sparse(csc_col_idx, csc_row_idx, csc_val, th.Size((A.size[1], A.size[0])))


def spadd(A, B):
  """Sum of sparse matrices"""
  rowC, colC, valC = spfuncs.SpAdd.apply(
      A.csr_row_idx, A.col_idx, A.val,
      B.csr_row_idx, B.col_idx, B.val,
      A.size)
  return Sparse(rowC, colC, valC, A.size)


def spmv(A, v):
  """Sparse matrix - dense vector product."""
  return spfuncs.SpMV.apply(A.csr_row_idx, A.col_idx, A.val, v, A.size)


def spmm(A, B):
  """Sparse matrix product."""
  rowC, colC, valC = spfuncs.SpMM.apply(
      A.csr_row_idx, A.col_idx, A.val, A.size,
      B.csr_row_idx, B.col_idx, B.val, B.size)
  sizeC = th.Size((A.size[0], B.size[1]))
  return Sparse(rowC, colC, valC, sizeC)


def sp_gram(s_mat):
  """A^T.A for A sparse"""
  raise NotImplemented


def sp_laplacian(s_mat):
  """diag(row_sum(A)) - A for A sparse"""
  # row_sum = spmv(A, )
  raise NotImplemented
