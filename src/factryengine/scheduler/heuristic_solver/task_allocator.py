from dataclasses import dataclass
from math import ceil

import numpy as np

from factryengine.models import Assignment, Resource, ResourceGroup

from .exceptions import AllocationError


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


class TaskAllocator:
    def allocate_task(
        self,
        resource_windows_dict: dict[int, np.array],
        assignments: list[Assignment],
        constraints: set[Resource],
        task_duration: float,
    ) -> dict[int, tuple[int, int]]:
        """
        allocates a task to the resources with the fastest completion based on the resource windows dict.
        Assignments determine which resources and how many.
        """

        # create base matrix
        resource_windows_matrix = self._create_matrix_from_resource_windows_dict(
            resource_windows_dict
        )

        if constraints:
            constraints_matrix = self.create_constraints_matrix(
                resource_constraints=constraints,
                resource_windows_matrix=resource_windows_matrix,
                task_duration=task_duration,
            )

            if assignments:
                # update the resource matrix with the constraint matrix
                self._apply_constraint_to_resource_windows_matrix(
                    constraints_matrix, resource_windows_matrix
                )
        else:
            constraints_matrix = None

        # build assignment matrices
        if assignments:
            assignments_matrix = self._create_assignments_matrix(
                assignments=assignments,
                resource_windows_matrix=resource_windows_matrix,
                task_duration=task_duration,
            )
        else:
            assignments_matrix = None

        # find the solution
        solution_matrix = self._solve_matrix(
            assignments_matrix=assignments_matrix,
            constraints_matrix=constraints_matrix,
            task_duration=task_duration,
        )

        # process solution to find allocated resource windows
        allocated_windows = self._get_resource_intervals(
            solution_matrix=solution_matrix,
        )

        return allocated_windows

    def _solve_matrix(
        self,
        task_duration: float,
        assignments_matrix: Matrix = None,
        constraints_matrix: Matrix = None,
    ) -> Matrix:
        """
        Takes the task matrix as input and finds the earliest solution
        where the work of the resources equals the task duration.
        If no solution is found, returns AllocationError.
        Returns a matrix with the solution as the last row.
        """

        # Check if total resources in assignments_matrix meet task_duration
        assignments_meet_duration = (
            np.sum(assignments_matrix.resource_matrix, axis=1) >= task_duration
            if assignments_matrix
            else True
        )

        # Check if first resource in constraints_matrix meets task_duration
        constraints_meet_duration = (
            np.sum(constraints_matrix.resource_matrix, axis=1) >= task_duration
            if constraints_matrix
            else True
        )

        # Combine conditions
        meets_duration_condition = assignments_meet_duration & constraints_meet_duration

        # Find index of first true condition
        solution_index = np.argmax(meets_duration_condition)

        # check if solution exists
        if solution_index == 0:
            raise AllocationError("No solution found.")

        # Choose the matrix to work with
        working_matrix = (
            assignments_matrix if assignments_matrix else constraints_matrix
        )

        # select the resources which are part of the solution
        solution_resources_mask = ~working_matrix.resource_matrix.mask[solution_index]
        solution_resource_ids = working_matrix.resource_ids[solution_resources_mask]

        end_index = solution_index + 1
        # filter resource matrix using solution resource mask
        solution_resource_matrix = working_matrix.resource_matrix[
            :end_index, solution_resources_mask
        ]

        # do linear regression to find precise solution
        # only use the last two rows of the matrix where
        # where the first row is the solution row
        last_two_rows = slice(-2, None)
        interval, solution_row = self._solve_task_end(
            resource_matrix=solution_resource_matrix[:][last_two_rows],
            intervals=working_matrix.intervals[:end_index][last_two_rows],
            task_duration=task_duration,
        )

        # update the solution row
        solution_resource_matrix[-1] = solution_row

        solution_intervals = np.append(
            working_matrix.intervals[:solution_index], interval
        )

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
    ) -> (int, np.array):
        """
        Calculates the end of a task given a resource matrix, intervals, and task duration.
        """

        # Initialize total slope and intercept
        total_slope, total_intercept = 0, 0

        # List to store slope and intercept values for each column
        slope_intercept_values = []

        # Calculate slope and intercept for each column
        for resource_col in range(resource_matrix.shape[1]):
            slope, intercept = np.polyfit(
                x=intervals, y=resource_matrix[:, resource_col], deg=1
            )
            total_slope += slope
            total_intercept += intercept
            slope_intercept_values.append((slope, intercept))

        # Compute the column 0 value that makes the sum of the predicted values equal to the task duration
        col0_value = (task_duration - total_intercept) / total_slope

        # Compute the corresponding values for the other columns
        other_columns_values = [
            slope * col0_value + intercept
            for slope, intercept in slope_intercept_values
        ]

        # Return a numpy array containing the solved value for column 0 and the predicted values for the other columns
        return col0_value, other_columns_values

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
    ) -> dict[int, tuple[int, int]]:
        """
        gets the resource intervals from the solution matrix.
        """
        end_index = solution_matrix.resource_matrix.shape[0] - 1

        resource_windows_dict = {}

        # loop through resource ids and resource intervals
        for resource_id, resource_intervals in zip(
            solution_matrix.resource_ids, solution_matrix.resource_matrix.T
        ):
            # ensure only continuous intervals are selected
            # TODO not working correctly, what if no more movement? use resource windows matrix instead
            start_index = self._find_last_zero_index(resource_intervals)

            if start_index is not None:
                resource_windows_dict[resource_id] = (
                    ceil(round(solution_matrix.intervals[start_index], 1)),
                    ceil(round(solution_matrix.intervals[end_index], 1)),
                )
        return resource_windows_dict

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

    def _cumsum_reset_at_minus_one_2d(self, arr: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self._cumsum_reset_at_minus_one, axis=0, arr=arr)

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
        """
        replaces masked values in an array with nan.
        """
        result_array = np.full(mask.shape, np.nan)  # Initialize with nan
        result_array[mask] = array
        return result_array

    def _create_matrix_from_resource_windows_dict(
        self, windows_dict: dict[int, np.ndarray]
    ) -> Matrix:
        """
        creates a matrix from a dictionary of resource windows.
        """
        # convert the dictionary of resource windows to a numpy array
        resource_windows_list = list(windows_dict.values())

        # get all window interval boundaries
        all_interval_boundaries = []
        for window in resource_windows_list:
            all_interval_boundaries.extend(window["start"].ravel())
            all_interval_boundaries.extend(window["end"].ravel())

        # find unique values
        intervals = np.unique(all_interval_boundaries)

        # first column is the interval boundaries
        matrix = [intervals]

        # loop through the resource windows and create a column for each resource
        for window in resource_windows_list:
            window_boundaries = np.dstack((window["start"], window["end"])).flatten()

            missing_boundaries_mask = np.isin(intervals, window_boundaries)

            window_durations = np.dstack(
                (window["is_split"], window["duration"])
            ).flatten()

            # window_durations_cumsum = self._cumsum_reset_at_minus_one(window_durations)

            resource_column = self._replace_masked_values_with_nan(
                window_durations, missing_boundaries_mask
            )

            matrix.append(resource_column)

        # create numpy matrix
        matrix = np.dstack(matrix)

        # fill nan values with linear interpolation
        matrix_nan_filled = self._interpolate_y_values(matrix)[0]

        # select only the resource columns
        resource_matrix = np.ma.MaskedArray(matrix_nan_filled[:, 1:])

        # extract intervals
        resource_ids = np.array(list(windows_dict.keys()))
        return Matrix(
            resource_ids=resource_ids,
            intervals=intervals,
            resource_matrix=resource_matrix,
        )

    def _create_resource_group_matrix(
        self,
        resource_group: ResourceGroup,
        resource_count: int,
        resource_windows_matrix: Matrix,
    ) -> Matrix:
        """
        Creates a resource group matrix from a resource group and resource windows matrix.
        """

        resource_ids = np.array(resource_group.get_resource_ids())

        # find the resources that exist in the windows matrix
        available_resources = np.intersect1d(
            resource_ids, resource_windows_matrix.resource_ids
        )

        # check if there are enough resources available
        if len(available_resources) < resource_count:
            raise AllocationError(
                f"Assignment {resource_group.id} requires {resource_count} "
                f"entities but only {len(available_resources)} are available."
            )

        # Find the indices of the available resources in the windows matrix
        resource_indexes = np.where(
            np.isin(resource_windows_matrix.resource_ids, available_resources)
        )[0]

        # Build the resource_matrix for the resource group matrix
        resource_matrix = resource_windows_matrix.resource_matrix[:, resource_indexes]

        resource_matrix_cumsum = self._cumsum_reset_at_minus_one_2d(resource_matrix)

        # mask all but the k largest elements per row
        resource_matrix_masked = self._mask_smallest_elements_except_top_k_per_row(
            resource_matrix_cumsum, resource_count
        )

        return Matrix(
            resource_ids=resource_ids,
            intervals=resource_windows_matrix.intervals,
            resource_matrix=resource_matrix_masked,
        )

    def create_constraints_matrix(
        self, resource_constraints, resource_windows_matrix, task_duration
    ) -> None:
        """
        Checks if the resource constraints are available and updates the resource windows matrix.
        """
        # get the constraint resource ids
        resource_ids = np.array([resource.id for resource in resource_constraints])

        # check if all resource constraints are available
        if not np.all(np.isin(resource_ids, resource_windows_matrix.resource_ids)):
            raise AllocationError("All resource constraints are not available")

        # Find the indices of the available resources in the windows matrix
        resource_indexes = np.where(
            np.isin(resource_windows_matrix.resource_ids, resource_ids)
        )[0]

        # get the windows for the resource constraints
        constraint_windows = resource_windows_matrix.resource_matrix[
            :, resource_indexes
        ]

        # Compute the minimum along axis 1, mask values <= 0, and compute the cumulative sum
        # devide by the number of resources to not increase the task completion time
        min_values_matrix = (
            np.min(constraint_windows, axis=1, keepdims=True)
            * np.ones_like(constraint_windows)
            / len(resource_ids)
        )

        resource_matrix = np.ma.masked_less_equal(
            x=min_values_matrix,
            value=0,
        ).cumsum(axis=0)

        # rasie error if no solution exists
        if not np.any(resource_matrix >= task_duration):
            raise AllocationError("No solution for the resource constraints")

        return Matrix(
            resource_ids=resource_ids,
            intervals=resource_windows_matrix.intervals,
            resource_matrix=resource_matrix,
        )

    def _apply_constraint_to_resource_windows_matrix(
        constraint_matrix: Matrix, resource_windows_matrix: Matrix
    ) -> None:
        """
        Adds reset to windows where the constraints are not available.
        Resets are represented by -1.
        """
        # create a mask from the constraint matrix
        mask = (
            np.ones_like(resource_windows_matrix.resource_matrix.data, dtype=bool)
            * constraint_matrix.resource_matrix.mask
        )
        # add reset to the resource matrix
        resource_windows_matrix.resource_matrix[mask] = -1

        # solution_intervals = resource_windows_matrix.intervals[
        #     constraint_matrix.flatten() >= task_duration
        # ]
        # # check if a solution exists else raise an error
        # if not solution_intervals:
        #     raise AllocationError("No solution for the resource constraints")

        # # update the resource matrix with the constraint matrix
        # # require reset when constraints are note available

        # mask = np.ones_like(resource_matrix.data, dtype=bool) * constraint_matrix.mask

        # resource_windows_matrix.resource_matrix[mask] = -1

        # return solution_intervals

    def _create_assignments_matrix(
        self,
        assignments: list[Assignment],
        resource_windows_matrix: Matrix,
        task_duration: int,
    ) -> Matrix:
        assignment_matrices = []
        for assignment in assignments:
            # create resource group matrices
            resource_group_matrices = []
            for resource_group in assignment.resource_groups:
                resource_group_matrix = self._create_resource_group_matrix(
                    resource_group=resource_group,
                    resource_count=assignment.resource_count,
                    resource_windows_matrix=resource_windows_matrix,
                )

                resource_group_matrices.append(resource_group_matrix)

            # keep resource group matrix rows with the fastest completion
            assignment_matrix = Matrix.compare_update_mask_and_merge(
                resource_group_matrices
            )
            assignment_matrices.append(assignment_matrix)

        # merge assignment matrices
        assignments_matrix = Matrix.merge(assignment_matrices)

        # check if solution exists
        if not np.any(assignments_matrix.resource_matrix >= task_duration):
            raise AllocationError("No solution for the resource constraints")

        return assignments_matrix
