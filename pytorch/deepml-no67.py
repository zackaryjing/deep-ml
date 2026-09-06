# problem: Implement Compressed Column Sparse Matrix Format (CSC)

import torch


def compressed_col_sparse_matrix(dense_matrix: torch.Tensor):
    """
    Convert a dense matrix tensor into its Compressed Column Sparse (CSC) representation
    using PyTorch's built-in sparse CSC tensor support.

    :param dense_matrix: 2D torch.Tensor representing the dense matrix
    :return: Tuple of (values, row_indices, col_pointer) as torch.Tensors
    """
    m = dense_matrix.to_sparse_csc()
    return m.values(),m.row_indices(),m.ccol_indices()


def main():
    dense_matrix = torch.tensor([
        [0, 0, 3, 0],
        [1, 0, 0, 4],
        [0, 2, 0, 0]
    ])
    vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)
    print(vals, row_idx, col_ptr)


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-06 14:57:25
#
