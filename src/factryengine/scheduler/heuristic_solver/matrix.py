from dataclasses import dataclass

import numpy as np


@dataclass
class Matrix:
    """
    Datastructure for representing resource windows as a matrix.
    Used in the allocation of tasks.
    Uses numpy arrays for efficient computation.
    """

    resource_ids: np.ndarray  # 1d array of resource ids
    intervals: np.ndarray  # 1d array of intervals
    resource_matrix: np.ma.core.MaskedArray  # 2d array of resource windows

    @classmethod
    def merge(cls, matrices: list["Matrix"]) -> "Matrix":
        """
        merges a list of matrices into one matrix.
        """
        resource_ids = np.concatenate([matrix.resource_ids for matrix in matrices])

        # Check if intervals are the same
        first_intervals = matrices[0].intervals
        if any(
            not np.array_equal(first_intervals, matrix.intervals) for matrix in matrices
        ):
            raise ValueError("All matrices must have the same intervals")

        resource_matrix = np.ma.hstack([matrix.resource_matrix for matrix in matrices])
        return cls(resource_ids, first_intervals, resource_matrix)

    @classmethod
    def compare_update_mask_and_merge(cls, matrices: list["Matrix"]) -> "Matrix":
        """
        Compares each row of each array in the list and masks the rows with smallest sums.
        Returns the combined array with the masked rows.
        """
        row_sums = [
            np.sum(matrix.resource_matrix, axis=1, keepdims=True) for matrix in matrices
        ]
        max_sum_index = np.argmax(np.hstack(row_sums), axis=1)

        # update the masks
        for i, matrix in enumerate(matrices):
            mask = np.ones_like(matrix.resource_matrix) * (max_sum_index[:, None] != i)
            matrix.resource_matrix.mask = mask

        return cls.merge(matrices)

    @classmethod
    def trim(cls, original_matrix: "Matrix", trim_matrix: "Matrix") -> "Matrix":
        """
        Trims a Matrix based on another
        """

        return cls(
            resource_ids=original_matrix.resource_ids,
            intervals=trim_matrix.intervals,
            resource_matrix=original_matrix.resource_matrix[
                : len(trim_matrix.intervals)
            ],
        )
