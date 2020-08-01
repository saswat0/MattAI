import torch
from torch.autograd import Function
from torch.autograd import Variable
from .._ext import sparse

class Coo2Csr(Function):
  @staticmethod
  def forward(ctx, row_idx, col_idx, val, size):
    ctx.size = size
    csr_row_idx = row_idx.new() 
    csr_col_idx = col_idx.new() 
    csr_val = val.new() 
    permutation = csr_row_idx.new()
    sparse.coo2csr(row_idx, col_idx, val, csr_row_idx, csr_col_idx, csr_val,
                   permutation, size[0], size[1])
    ctx.permutation = permutation
    return (csr_row_idx, csr_col_idx, csr_val)

  @staticmethod
  def backward(ctx, grad_csr_row_idx, grad_csr_col_idx,
               grad_csr_val):
    grad_size = None
    grad_row_idx = None
    grad_col_idx = None

    permutation = ctx.permutation

    grad_val = grad_csr_val[permutation.long()]

    return (grad_row_idx, grad_col_idx, grad_val, grad_size)


class Transpose(Function):
  @staticmethod
  def forward(ctx, row_idx, col_idx, val, size):
    csc_row_idx = row_idx.new()
    csc_col_idx = row_idx.new()
    csc_val = val.new()
    sparse.csr2csc(row_idx, col_idx, val, csc_row_idx, csc_col_idx, csc_val, size[0], size[1])
    # csc_row_idx = Variable(csc_row_idx)
    # csc_col_idx = Variable(csc_col_idx)
    # csc_val = Variable(csc_val)

    ctx.save_for_backward(csc_row_idx, csc_col_idx, val)
    ctx.size = size

    return (csc_row_idx, csc_col_idx, csc_val)

  @staticmethod
  def backward(ctx, grad_csc_row_idx, grad_csc_col_idx, grad_csc_val):
    csc_row_idx, csc_col_idx, val = ctx.saved_variables
    size = ctx.size
    row_idx = csc_row_idx.data.new()
    col_idx = csc_col_idx.data.new()
    grad_val = grad_csc_val.data.new()
    sparse.csr2csc(
        csc_col_idx.data, csc_row_idx.data, grad_csc_val.data, col_idx, row_idx, grad_val, size[1], size[0])
    grad_col_idx = None
    grad_row_idx = None
    grad_val = Variable(grad_val)
    return (grad_row_idx, grad_col_idx, grad_val, None)


class SpAdd(Function):
  """Sum of sparse matrices."""

  @staticmethod
  def forward(ctx, rowA, colA, valA, rowB, colB, valB, size, alpha=1.0, beta=1.0):
    ctx.matrix_size = size
    ctx.alpha = alpha
    ctx.beta = beta

    rowC = torch.IntTensor().cuda()
    colC = torch.IntTensor().cuda()
    valC = torch.FloatTensor().cuda()
    sparse.spadd_forward(
        rowA, colA, valA, 
        rowB, colB, valB, 
        rowC, colC, valC, 
        alpha, beta,
        size[0], size[1])

    ctx.save_for_backward(rowA, colA, rowB, colB, rowC, colC)
    return rowC, colC, valC

  @staticmethod
  def backward(ctx, grad_rowC, grad_colC, grad_valC):
    rowA, colA, rowB, colB, rowC, colC = ctx.saved_variables
    size = ctx.matrix_size
    alpha = ctx.alpha
    beta = ctx.beta

    grad_rowA = grad_colA = grad_valA = None
    grad_rowB = grad_colB = grad_valB = None
    grad_size = None
    grad_alpha = None
    grad_beta = None

    # dL/dA_ik = dL/dC_ik*dC_ik/dA_ik where
    # dC_ik/dA_ik = 1 iff A_ik != 0
    # dL/dA_ik should select in dL/dC_ik following A's sparsity pattern
    grad_valA = grad_valC.data.new()
    grad_valB = grad_valC.data.new()
    sparse.spadd_backward(
        rowA.data, colA.data, grad_valA,
        rowB.data, colB.data, grad_valB,
        rowC.data, colC.data, grad_valC.data,
        alpha, beta, size[0], size[1])

    grad_valA = Variable(grad_valA)
    grad_valB = Variable(grad_valB)

    return grad_rowA, grad_colA, grad_valA, \
           grad_rowB, grad_colB, grad_valB, grad_size, grad_alpha, grad_beta


class SpMV(Function):
  """Sparse matrix-vector product."""

  @staticmethod
  def forward(ctx, row, col, val, vector, size):
    ctx.save_for_backward(row, col, val, vector)
    ctx.matrix_size = size
    output = vector.new() 
    sparse.spmv(
        row, col, val, 
        vector, output,
        size[0], size[1], False)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    row, col, val, vector = ctx.saved_variables
    size = ctx.matrix_size
    grad_row = grad_col = grad_val = None
    grad_vector = None
    grad_size = None

    grad_vector = vector.data.new()
    sparse.spmv(
        row.data, col.data, val.data, 
        grad_output.data, grad_vector,
        size[0], size[1], True)

    grad_val = val.data.new()
    sparse.spmv_backward_matrix(
        row.data, col.data, 
        vector.data, grad_output.data, grad_val,
        size[0], size[1])

    grad_vector = Variable(grad_vector)
    grad_val = Variable(grad_val)

    return grad_row, grad_col, grad_val, grad_vector, grad_size


class SpMM(Function):
  """Product of sparse matrices."""

  @staticmethod
  def forward(ctx, rowA, colA, valA, sizeA, rowB, colB, valB, sizeB):
    ctx.A_size = sizeA
    ctx.B_size = sizeB
    rowC = torch.IntTensor().cuda()
    colC = torch.IntTensor().cuda()
    valC = torch.FloatTensor().cuda()
    sparse.spmm_forward(
        rowA, colA, valA, sizeA[0], sizeA[1],
        rowB, colB, valB, sizeB[0], sizeB[1],
        rowC, colC, valC)
    ctx.save_for_backward(rowA, colA, valA, rowB, colB, valB, rowC, colC)
    return rowC, colC, valC

  @staticmethod
  def backward(ctx, grad_rowC, grad_colC, grad_valC):
    rowA, colA, valA, rowB, colB, valB, rowC, colC = ctx.saved_variables
    sizeA = ctx.A_size
    sizeB = ctx.B_size
    grad_rowA = grad_colA = grad_valA = None
    grad_rowB = grad_colB = grad_valB = None
    grad_sizeA = None
    grad_sizeB = None

    grad_valA = valA.data.new()
    grad_valB = valB.data.new()

    sparse.spmm_backward(
        rowA.data, colA.data, valA.data, grad_valA, sizeA[0], sizeA[1],
        rowB.data, colB.data, valB.data, grad_valB, sizeB[0], sizeB[1],
        rowC.data, colC.data, grad_valC.data)

    grad_valA = Variable(grad_valA)
    grad_valB = Variable(grad_valB)

    return grad_rowA, grad_colA, grad_valA, grad_sizeA, grad_rowB, grad_colB, grad_valB, grad_sizeB
