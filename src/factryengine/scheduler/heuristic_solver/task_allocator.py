from collections import namedtuple
from itertools import compress

import numpy as np

from factryengine.models import Assignment

from .exceptions import AllocationError

Matrix = namedtuple("Matrix", ["resource_ids", "matrix"])


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
        # initialize task allocated windows
        task_allocated_windows = {}

        for assignment in assignments:
            # create resource matrix
            resource_matrix = self._create_resource_coordinates_matrix(
                resource_windows_dict
            )

            # create assignment matrix from resource matrix
            assignment_matrix = self._create_assignment_matrix(
                assignment, resource_matrix
            )

            # find solution using assignment matrix
            solution_matrix, solution_resource_ids = self._solve_matrix(
                matrix=assignment_matrix,
                task_duration=task_duration,
                resource_count=assignment.entity_count,
            )

            # get allocated windows
            assignment_allocated_windows = self._get_resource_intervals(
                solution_matrix, solution_resource_ids, resource_matrix
            )

            # update task allocated windows
            task_allocated_windows.update(assignment_allocated_windows)

            # check if not last iteration
            if assignment != assignments[-1]:
                # update remove resource ids from resource windows dict
                solution_resource_ids_flattend = self._flatten_list_of_tuples(
                    solution_resource_ids
                )
                resource_windows_dict = self._update_resource_windows_dict(
                    resource_windows_dict, solution_resource_ids_flattend
                )
        return task_allocated_windows

    def _solve_matrix(
        self,
        matrix: Matrix,
        task_duration: float,
        resource_count: int,
    ) -> (dict[int, np.array], list[tuple[int]]):
        """
        Finds the earliest possible solution for a given task based on its duration and
        the number of resources available. The method uses a matrix representation of
        the resource windows to calculate the optimal allocation of the task.
        """
        resource_ids = matrix.resource_ids
        matrix = matrix.matrix

        resource_matrix = matrix[:, 1:]

        # check if task duration is greater than the maximum resource duration
        if resource_count == 1 and task_duration > resource_matrix.max():
            raise AllocationError(
                f"Task duration {task_duration} is greater than the maximum resource "
                f"duration {resource_matrix.max()}."
            )

        # mask all but the k largest elements per row
        masked_resource_matrix = self._mask_smallest_elements_except_top_k_per_row(
            resource_matrix, resource_count
        )

        arr_sum = np.sum(masked_resource_matrix, axis=1)

        # check if task duration is greater than the maximum resource duration
        if task_duration > arr_sum.max():
            raise AllocationError(
                f"Task duration {task_duration} is greater than the maximum resource "
                f"duration {arr_sum.max()}."
            )

        # get solution index and resource ids
        solution_index = np.argmax(arr_sum >= task_duration)
        solution_resources_mask = ~masked_resource_matrix.mask[solution_index]
        # filter resource ids using solution resource mask
        solution_resource_ids = list(compress(resource_ids, solution_resources_mask))

        # solve matrix
        # build matrix which has the solution

        # include first column in solution
        solution_cols_mask = np.concatenate([[True], solution_resources_mask])

        # select solution data + 1 row to do linear regression
        solution_matrix = matrix[: solution_index + 1, solution_cols_mask]

        # do linear regression to find precise solution
        solution = self._solve_task_end(solution_matrix[-2:], task_duration)

        # add solution to solution matrix
        solution_matrix = np.vstack(
            (solution_matrix[:solution_index], np.atleast_2d(solution))
        )
        return solution_matrix, solution_resource_ids

    def _solve_task_end(self, matrix: np.array, task_duration: int) -> np.array:
        # Calculate slopes and intercepts for all columns after the first one directly
        # into total_m and total_b
        total_m, total_b = 0, 0
        mb_values = []
        for i in range(1, matrix.shape[1]):
            m, b = np.polyfit(matrix[:, 0], matrix[:, i], 1)
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
        result = np.array([col0] + other_cols)

        return result

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
        resource_ids: list[tuple[int]],
        resource_matrix: Matrix,
    ) -> dict[int, tuple[float, float]]:
        """
        gets the resource intervals from the solution matrix.
        """
        end_index = solution_matrix.shape[0] - 1

        resource_windows_dict = {}

        for group_resource_ids in resource_ids:
            for resource_id in group_resource_ids:
                resource_column = resource_matrix.resource_ids.index(resource_id) + 1
                resource_durations = resource_matrix.matrix[
                    : end_index + 1, resource_column
                ].T
                start_index = self._find_last_zero_index(resource_durations)

                if start_index is not None:
                    resource_windows_dict[resource_id] = (
                        solution_matrix[start_index, 0],
                        solution_matrix[end_index, 0],
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
        self, array, k
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
        masked_array = np.ma.masked_array(array, mask=mask)
        return masked_array

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

    def _create_resource_coordinates_matrix(
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
        return Matrix(
            resource_ids=list(windows_dict.keys()), matrix=coordinates_nan_filled[0]
        )

    def _create_assignment_matrix(
        self, assignment: Assignment, matrix: Matrix
    ) -> Matrix:
        """Create the matrices for the resource groups in the assignment"""

        matrix_durations = matrix.matrix[:, 1:]
        output_matrix = matrix.matrix[:, [0]]
        resource_ids: list[tuple[int]] = []

        for group_resource_ids in assignment.get_resource_ids():
            # Check if the team resources exist in the matrix resource ids else skip
            if not set(group_resource_ids).issubset(set(matrix.resource_ids)):
                continue

            # Get the indices of the team resources in the matrix resource ids
            columns = [
                matrix.resource_ids.index(resource_id)
                for resource_id in group_resource_ids
            ]

            group_matrix = matrix_durations[:, columns]

            # Sum the group matrix if there are multiple resources in the group
            if len(group_resource_ids) > 1:
                group_matrix = np.sum(group_matrix, axis=1, keepdims=True)

            # Add the team matrix to the output matrix
            output_matrix = np.hstack((output_matrix, group_matrix))

            resource_ids.append(group_resource_ids)

        # check if the required entity count is met
        if assignment.entity_count > output_matrix.shape[1] - 1:
            raise AllocationError(
                f"Assignment {assignment.id} requires {assignment.entity_count} "
                f"entities but only {output_matrix.shape[1] - 1} are available."
            )

        return Matrix(resource_ids=resource_ids, matrix=output_matrix)

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
