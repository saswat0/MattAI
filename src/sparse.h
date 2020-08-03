int coo2csr(THCudaIntTensor *row_idx, 
            THCudaIntTensor *col_idx,
            THCudaTensor *val,
            THCudaIntTensor *csr_row_idx,
            THCudaIntTensor *csr_col_idx,
            THCudaTensor *csr_val,
            THCudaIntTensor *permutation,
            const long rows, const long cols);

int csr2csc(THCudaIntTensor *row_idx, 
            THCudaIntTensor *col_idx,
            THCudaTensor *val,
            THCudaIntTensor *csc_row_idx, 
            THCudaIntTensor *csc_col_idx,
            THCudaTensor *csc_val,
            const long rows, const long cols);

int spadd_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *A_val,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *B_val,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_val,
    const float alpha, const float beta, const long rows, const long cols);

int spadd_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *gradA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *gradB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *gradC,
    const float alpha, const float beta, const long rows, const long cols);


int spmv(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col, THCudaTensor *val,
    THCudaTensor *vector,
    THCudaTensor *output,
    const long rows, const long cols, const int transpose);


int spmv_backward_matrix(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col,
    THCudaTensor *vector,
    THCudaTensor *grad_output,
    THCudaTensor *grad_matrix,
    const long rows, const long cols);

int spmm_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *A_val,
    const long rowsA, const long colsA, 
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *B_val,
    const long rowsB, const long colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_val);

int spmm_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col,
    THCudaTensor *A_val, THCudaTensor *A_grad_val,
    const long rowsA, const long colsA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col,
    THCudaTensor *B_val, THCudaTensor *B_grad_val,
    const long rowsB, const long colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_grad_val);

