# problem: Implement Compressed Row Sparse Matrix (CSR) Format Conversion

import torch


def compressed_row_sparse_matrix(dense_matrix) -> tuple:
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation
    using PyTorch's built-in sparse CSR tensor support.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values tensor, column indices tensor, row pointer tensor)
    """
    matrix = torch.tensor(dense_matrix)
    csr_m = matrix.to_sparse_csr()
    return csr_m.values().tolist(), csr_m.col_indices().tolist(), csr_m.crow_indices().tolist()


def main():
    dense_matrix = [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [3, 0, 4, 0],
        [1, 0, 0, 5]
    ]
    vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
    print("Values array:", vals)
    print("Column indices array:", col_idx)
    print("Row pointer array:", row_ptr)


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-06 14:06:13
#
