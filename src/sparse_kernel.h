#ifndef SPARSE_KERNEL_H_FTUJX27W
#define SPARSE_KERNEL_H_FTUJX27W

#ifdef __cplusplus
extern "C" {
#endif

void spmv_backward_matrix_cuda(
    const int* p_cooRow, const int* p_csrCol, const float* p_vector,
    const float* p_grad_output, float* p_grad_matrix,
    const int rows, const int cols, const int nnz);

void spadd_backward_cuda(
      const int* p_csr_rowA, const int* p_csr_colA, float* p_gradA, const int nnzA,
      const int* p_csr_rowB, const int* p_csr_colB, float* p_gradB, const int nnzB,
      const int* p_coo_rowC, const int* p_csr_colC, const float* p_gradC, const int nnzC,
      const float alpha, const float beta, const int rows, const int cols);

void matmul_preserve_sparsity_cuda(
      const int* p_csr_row1, const int* p_csr_col1, const float* p_data1,
      const int* p_csr_row2, const int* p_csr_col2, const float* p_data2,
      const int* p_coo_row_out, const int* p_coo_col_out, float* p_out,
      const int nnz_out);

#ifdef __cplusplus
}
#endif

#endif /* end of include guard: SPARSE_KERNEL_H_FTUJX27W */
