from dataclasses import dataclass
from math import ceil

import numpy as np

from factryengine.models import Assignment, ResourceGroup

from .exceptions import AllocationError


# matrix dataclass
@dataclass
class Matrix:
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


class TaskAllocator:
    def allocate_task(
        self,
        resource_windows_dict: dict[int, np.array],
        assignments: list[Assignment],
        task_duration: float,
    ) -> dict[int, tuple[float, float]]:
        """
        allocates a task to the resources with the fastest completion based on the resource windows dict.
        Assignments determine which resources and how many.
        """

        # create resource matrix
        base_resource_matrix = self._create_base_resource_matrix(resource_windows_dict)

        assignment_matrices = []
        for assignment in assignments:
            resource_group_matrices = []
            for resource_group in assignment.resource_groups:
                resource_group_matrix = self._create_resource_group_matrix(
                    resource_group=resource_group,
                    resource_count=assignment.resource_count,
                    matrix=base_resource_matrix,
                )

                resource_group_matrices.append(resource_group_matrix)
            assignment_matrix = Matrix.compare_update_mask_and_merge(
                resource_group_matrices
            )
            assignment_matrices.append(assignment_matrix)

        task_matrix = Matrix.merge(assignment_matrices)
        solution_matrix = self._solve_matrix(task_matrix, task_duration)

        allocated_windows = self._get_resource_intervals(
            solution_matrix=solution_matrix, resource_matrix=base_resource_matrix
        )

        return allocated_windows

    def _solve_matrix(
        self,
        matrix: Matrix,
        task_duration: float,
    ) -> (dict[int, np.array], list[tuple[int]]):
        """
        Finds the earliest possible solution for a given task based on its duration and
        the number of resources available. The method uses a matrix representation of
        the resource windows to calculate the optimal allocation of the task.
        """

        # check if task duration is greater than the maximum resource duration
        # if resource_count == 1 and task_duration > matrix.resource_matrix.max():
        #     raise AllocationError(
        #         f"Task duration {task_duration} is greater than the maximum resource "
        #         f"duration {matrix.resource_matrix.max()}."
        #     )

        row_sums = np.sum(matrix.resource_matrix, axis=1)

        # check if task duration is greater than the maximum resource duration
        if task_duration > row_sums.max():
            raise AllocationError(
                f"Task duration {task_duration} is greater than the maximum resource "
                f"duration {row_sums.max()}."
            )

        # get solution index and resource ids
        solution_index = np.argmax(row_sums >= task_duration)
        solution_resources_mask = ~matrix.resource_matrix.mask[solution_index]
        # filter resource ids using solution resource mask
        solution_resource_ids = matrix.resource_ids[solution_resources_mask]

        # solve matrix
        # build matrix which has the solution

        # include first column in solution
        # solution_cols_mask = np.concatenate([[True], solution_resources_mask])

        # select solution data + 1 row to do linear regression
        solution_resource_matrix = matrix.resource_matrix[
            : solution_index + 1, solution_resources_mask
        ]

        # do linear regression to find precise solution
        interval, solution_row = self._solve_task_end(
            resource_matrix=solution_resource_matrix[-2:],
            intervals=matrix.intervals[: solution_index + 1][-2:],
            task_duration=task_duration,
        )

        # add solution to solution matrix
        solution_resource_matrix = np.vstack(
            (solution_resource_matrix[:solution_index], np.atleast_2d(solution_row))
        )
        solution_intervals = np.append(matrix.intervals[:solution_index], interval)

        return Matrix(
            resource_ids=solution_resource_ids,
            intervals=solution_intervals,
            resource_matrix=solution_resource_matrix,
        )

    def _solve_task_end(
        self,
        resource_matrix: np.ma.MaskedArray,
        intervals: np.ndarray,
        task_duration: int,
    ) -> np.array:
        # Calculate slopes and intercepts for all columns after the first one directly
        # into total_m and total_b
        total_m, total_b = 0, 0
        mb_values = []
        for i in range(0, resource_matrix.shape[1]):
            m, b = np.polyfit(x=intervals, y=resource_matrix[:, i], deg=1)
            total_m += m
            total_b += b
            mb_values.append((m, b))

        # Compute the column 0 value that makes the sum of the predicted values equal to
        # the desired sum
        col0 = (task_duration - total_b) / total_m

        # Compute the corresponding values for the other columns using matrix operations
        other_cols = [m * col0 + b for m, b in mb_values]

        # Return a numpy array containing the solved value for column 0 and the
        # predicted values for the other columns
        return col0, other_cols

    def _find_last_zero_index(self, arr):
        """
        returns the index of the last zero in an array.
        """

        # if last element is zero return None
        if arr[-1] == 0:
            return None

        # find the indices of zeros from the end
        zero_indices = np.nonzero(arr == 0)  # Find the indices of zeros from the end

        if zero_indices[0].size > 0:
            return zero_indices[0][-1]
        else:
            return 0

    def _get_resource_intervals(
        self,
        solution_matrix: np.array,
        resource_matrix: Matrix,
    ) -> dict[int, tuple[float, float]]:
        """
        gets the resource intervals from the solution matrix.
        """
        end_index = solution_matrix.resource_matrix.shape[0] - 1

        resource_windows_dict = {}

        for resource_id in solution_matrix.resource_ids:
            resource_column = (resource_matrix.resource_ids == resource_id).argmax()
            resource_durations = resource_matrix.resource_matrix[:, resource_column].T
            start_index = self._find_last_zero_index(resource_durations)

            if start_index is not None:
                resource_windows_dict[resource_id] = (
                    ceil(round(solution_matrix.intervals[start_index], 1)),
                    ceil(round(solution_matrix.intervals[end_index], 1)),
                )

        return resource_windows_dict

    def find_task_start(
        self, resource_matrix: Matrix, resource_id: int, start_index, end_index
    ):
        resource_column = (
            Matrix.resource_ids.index(resource_id) + 1
        )  # add 1 because the first column is duration
        resource_matrix = resource_matrix[start_index:end_index, [0, resource_column]]
        # find the last zero index
        last_zero_index = self._find_last_zero_index(resource_matrix[:, 1])
        return

    def _mask_smallest_elements_except_top_k_per_row(
        self, array: np.ma.core.MaskedArray, k
    ) -> np.ma.core.MaskedArray:
        """
        Masks the smallest elements in an array, except for the k largest elements on
        each row. This is a helper method used in the finding of the earliest solution.
        Zeroes are also masked as they are not valid solutions.
        """
        indices = np.argpartition(array, -k, axis=1)
        mask = np.ones_like(array, dtype=bool)
        rows = np.arange(array.shape[0])[:, np.newaxis]
        mask[rows, indices[:, -k:]] = False
        mask[array == 0] = True
        array.mask = mask
        return array

    def _cumsum_reset_at_minus_one(self, a: np.ndarray) -> np.ndarray:
        """
        Computes the cumulative sum of an array but resets the sum to zero whenever a
        -1 is encountered. This is a helper method used in the creation of the resource
        windows matrix.
        """
        reset_at = a == -1
        a[reset_at] = 0
        without_reset = a.cumsum()
        overcount = np.maximum.accumulate(without_reset * reset_at)
        return without_reset - overcount

    def _interpolate_y_values(self, array: np.ndarray) -> np.ndarray:
        """
        Perform linear interpolation to fill nan values in multiple y-columns of the array,
        using advanced NumPy features without an explicit for-loop.

        :param array: A numpy array with shape (1, n, m) where the first column represents x-values
                    and the remaining columns represent multiple y-values, which may contain nan.
        :return: A numpy array with the same shape, where nan values in the y-columns have been
                    interpolated based on the x and y values of the non-nan points.
        """
        # Extracting x column and y-columns
        x = array[0, :, 0]
        y_columns = array[0, :, 1:]

        # Mask for nan values in y-columns
        nan_mask = np.isnan(y_columns)

        # Performing linear interpolation for each y-column
        for i in range(y_columns.shape[1]):
            # Mask and values for the current y-column
            current_y = y_columns[:, i]
            current_nan_mask = nan_mask[:, i]

            # Interpolating only the nan values
            current_y[current_nan_mask] = np.interp(
                x[current_nan_mask], x[~current_nan_mask], current_y[~current_nan_mask]
            )

        # Updating the y-columns in the array
        array[0, :, 1:] = y_columns

        return array

    def _replace_masked_values_with_nan(
        self, array: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        result_array = np.full(mask.shape, np.nan)  # Initialize with nan
        result_array[mask] = array
        return result_array

    def _create_base_resource_matrix(
        self, windows_dict: dict[int, np.ndarray]
    ) -> Matrix:
        windows = np.array(list(windows_dict.values()))
        boundaries = np.unique(np.dstack((windows["start"], windows["end"])))
        columns = [boundaries]
        for window in windows:
            window_boundaries = np.dstack((window["start"], window["end"])).flatten()

            missing_boundaries_mask = np.isin(boundaries, window_boundaries)

            window_durations = np.dstack(
                (window["is_split"], window["duration"])
            ).flatten()

            window_durations_cumsum = self._cumsum_reset_at_minus_one(window_durations)

            duration_column = self._replace_masked_values_with_nan(
                window_durations_cumsum, missing_boundaries_mask
            )

            columns.append(duration_column)

        coordinates = np.dstack((columns))
        coordinates_nan_filled = self._interpolate_y_values(coordinates)
        final = np.ma.MaskedArray(coordinates_nan_filled[0])
        resource_ids = np.array(list(windows_dict.keys()))
        return Matrix(
            resource_ids=resource_ids,
            intervals=final[:, 0].data,
            resource_matrix=final[:, 1:],
        )

    def _create_resource_group_matrix(
        self,
        resource_group: ResourceGroup,
        resource_count: int,
        matrix: Matrix,
    ) -> Matrix:
        """Create the matrices for the resource groups in the assignment"""

        resource_ids = np.array(resource_group.get_resource_ids())

        # find the resources that exist in the resource matrix
        available_resources = np.intersect1d(resource_ids, matrix.resource_ids)
        if len(available_resources) < resource_count:
            raise AllocationError(
                f"Assignment {resource_group.id} requires {resource_count} "
                f"entities but only {len(available_resources)} are available."
            )

        # Find the indices of the available resources in the resource matrix
        columns = np.where(np.isin(resource_ids, available_resources))[0]

        # Build the resource_matrix for the resource group matrix
        resource_matrix = matrix.resource_matrix[:, columns]

        # mask all but the k largest elements per row
        resource_matrix_masked = self._mask_smallest_elements_except_top_k_per_row(
            resource_matrix, resource_count
        )

        return Matrix(
            resource_ids=resource_ids,
            intervals=matrix.intervals,
            resource_matrix=resource_matrix_masked,
        )

    def _update_resource_windows_dict(
        self, resource_windows_dict: dict, solution_resource_ids: list
    ) -> dict:
        return {
            resource_id: resource_windows_dict[resource_id]
            for resource_id in resource_windows_dict.keys()
            if resource_id not in solution_resource_ids
        }

    def _flatten_list_of_tuples(self, list_of_tuples):
        return [item for sublist in list_of_tuples for item in sublist]

    def _compare_and_mask_arrays(
        self, arrays: list[np.ndarray]
    ) -> np.ma.core.MaskedArray:
        """
        Compares each row of each array in the list and masks the rows with smallest sums.
        Returns the combined array with the masked rows.
        """
        sums = [np.sum(arr, axis=1, keepdims=True) for arr in arrays]
        max_sum_index = np.argmax(np.hstack(sums), axis=1)

        masked_arrays = []
        for i, arr in enumerate(arrays):
            mask = np.ones_like(arr) * (max_sum_index[:, None] != i)
            masked_arrays.append(np.ma.masked_array(arr, mask=mask))

        return np.ma.hstack(masked_arrays)
