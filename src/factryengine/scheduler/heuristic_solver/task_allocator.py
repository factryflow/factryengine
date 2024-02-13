from math import ceil

import numpy as np

from factryengine.models import Assignment, Resource, ResourceGroup

from .exceptions import AllocationError
from .matrix import Matrix


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

        resource_windows_matrix = self._create_matrix_from_resource_windows_dict(
            resource_windows_dict
        )

        # create constraints matrix
        constraints_matrix = self._create_constraints_matrix(
            resource_constraints=constraints,
            resource_windows_matrix=resource_windows_matrix,
            task_duration=task_duration,
        )
        if assignments and constraints:
            # update the resource matrix with the constraint matrix
            self._apply_constraint_to_resource_windows_matrix(
                constraints_matrix, resource_windows_matrix
            )

        # build assignment matrices
        assignments_matrix = self._create_assignments_matrix(
            assignments=assignments,
            resource_windows_matrix=resource_windows_matrix,
            task_duration=task_duration,
        )

        # matrix to solve
        matrix_to_solve = assignments_matrix or constraints_matrix

        # find the solution
        solution_matrix = self._solve_matrix(
            matrix=matrix_to_solve,
            task_duration=task_duration,
        )
        # process solution to find allocated resource windows
        allocated_windows = self._get_resource_intervals(
            matrix=solution_matrix,
        )

        # add constraints to allocated windows
        if constraints and assignments:
            constraints_matrix_trimmed = Matrix.trim(
                original_matrix=constraints_matrix, trim_matrix=solution_matrix
            )
            allocated_windows.update(
                self._get_resource_intervals(
                    matrix=constraints_matrix_trimmed,
                )
            )

        return allocated_windows

    def _solve_matrix(
        self,
        task_duration: float,
        matrix: Matrix = None,
    ) -> Matrix:
        """
        Takes the task matrix as input and finds the earliest solution
        where the work of the resources equals the task duration.
        If no solution is found, returns AllocationError.
        Returns a matrix with the solution as the last row.
        """

        # Check if total resources in assignments_matrix meet task_duration

        matrix_meet_duration = np.sum(matrix.resource_matrix, axis=1) >= task_duration

        # Find index of first true condition
        solution_index = np.argmax(matrix_meet_duration)

        # check if solution exists
        if solution_index == 0:
            raise AllocationError("No solution found.")

        # select the resources which are part of the solution
        solution_resources_mask = ~matrix.resource_matrix.mask[solution_index]
        solution_resource_ids = matrix.resource_ids[solution_resources_mask]

        end_index = solution_index + 1
        # filter resource matrix using solution resource mask
        solution_resource_matrix = matrix.resource_matrix[
            :end_index, solution_resources_mask
        ]
        # do linear regression to find precise solution
        # only use the last two rows of the matrix where
        # where the first row is the solution row
        last_two_rows = slice(-2, None)
        interval, solution_row = self._solve_task_end(
            resource_matrix=solution_resource_matrix[:][last_two_rows],
            intervals=matrix.intervals[:end_index][last_two_rows],
            task_duration=task_duration,
        )

        # update the solution row
        solution_resource_matrix[-1] = solution_row

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
    ) -> tuple[int, np.array]:
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

    def _get_resource_intervals(
        self,
        matrix: np.array,
    ) -> dict[int, tuple[int, int]]:
        """
        gets the resource intervals from the solution matrix.
        """
        end_index = matrix.resource_matrix.shape[0] - 1
        resource_windows_dict = {}
        # loop through resource ids and resource intervals
        for resource_id, resource_intervals in zip(
            matrix.resource_ids, matrix.resource_matrix.T
        ):
            # ensure only continuous intervals are selected
            indexes = self._find_indexes(resource_intervals.data)
            if indexes is not None:
                start_index, end_index = indexes
                resource_windows_dict[resource_id] = (
                    ceil(round(matrix.intervals[start_index], 1)),
                    ceil(round(matrix.intervals[end_index], 1)),
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

            # replace masked values with nan

            resource_column = self._replace_masked_values_with_nan(
                window_durations, missing_boundaries_mask
            )

            # fill nan values with linear interpolation

            resource_column = self._linear_interpolate_nan(resource_column, intervals)

            # distribute the window durations over the intervals
            resource_column = self._diff_and_zero_negatives(resource_column)

            matrix.append(resource_column)

        # create numpy matrix

        matrix = np.stack(matrix, axis=1)

        # select only the resource columns
        resource_matrix = np.ma.MaskedArray(matrix[:, 1:])

        # extract intervals
        resource_ids = np.array(list(windows_dict.keys()))
        return Matrix(
            resource_ids=resource_ids,
            intervals=intervals,
            resource_matrix=resource_matrix,
        )

    def _diff_and_zero_negatives(self, arr):
        arr = np.diff(arr, prepend=0)
        # replace negative values with 0
        arr[arr < 0] = 0
        return arr

    def _create_resource_group_matrix(
        self,
        resource_group: ResourceGroup,
        resource_count: int,
        use_all_resources: bool,
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
        available_resources_count = len(available_resources)
        if available_resources_count == 0:
            return None

        # Find the indices of the available resources in the windows matrix
        resource_indexes = np.where(
            np.isin(resource_windows_matrix.resource_ids, available_resources)
        )[0]

        # Build the resource_matrix for the resource group matrix
        resource_matrix = resource_windows_matrix.resource_matrix[:, resource_indexes]
        # compute the cumulative sum of the resource matrix columns
        resource_matrix = self._cumsum_reset_at_minus_one_2d(resource_matrix)

        # mask all but the k largest elements per row
        if use_all_resources is False:
            if resource_count < available_resources_count:
                resource_matrix = self._mask_smallest_elements_except_top_k_per_row(
                    resource_matrix, resource_count
                )

        return Matrix(
            resource_ids=resource_ids,
            intervals=resource_windows_matrix.intervals,
            resource_matrix=resource_matrix,
        )

    def _create_constraints_matrix(
        self,
        resource_constraints: set[Resource],
        resource_windows_matrix: Matrix,
        task_duration: int,
    ) -> Matrix:
        """
        Checks if the resource constraints are available and updates the resource windows matrix.
        """
        if not resource_constraints:
            return None

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

        return Matrix(
            resource_ids=resource_ids,
            intervals=resource_windows_matrix.intervals,
            resource_matrix=resource_matrix,
        )

    def _apply_constraint_to_resource_windows_matrix(
        self, constraint_matrix: Matrix, resource_windows_matrix: Matrix
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

    def _create_assignments_matrix(
        self,
        assignments: list[Assignment],
        resource_windows_matrix: Matrix,
        task_duration: int,
    ) -> Matrix:
        if assignments == []:
            return None

        assignment_matrices = []
        for assignment in assignments:
            # create resource group matrices
            resource_group_matrices = []
            for resource_group in assignment.resource_groups:
                resource_group_matrix = self._create_resource_group_matrix(
                    resource_group=resource_group,
                    resource_count=assignment.resource_count,
                    use_all_resources=assignment.use_all_resources,
                    resource_windows_matrix=resource_windows_matrix,
                )
                if resource_group_matrix is None:
                    continue

                resource_group_matrices.append(resource_group_matrix)

            if resource_group_matrices == []:
                raise AllocationError("No resource groups with available resources.")

            # keep resource group matrix rows with the fastest completion
            assignment_matrix = Matrix.compare_update_mask_and_merge(
                resource_group_matrices
            )
            assignment_matrices.append(assignment_matrix)

        # merge assignment matrices
        assignments_matrix = Matrix.merge(assignment_matrices)

        # check if solution exists
        if not np.any(assignments_matrix.resource_matrix >= task_duration):
            raise AllocationError("No solution found.")

        return assignments_matrix

    def _find_indexes(self, arr: np.array) -> tuple[int, int] | None:
        """
        Find the start and end indexes from the last zero to the last number with no increase in a NumPy array.
        """
        # if last element is zero return None
        if arr[-1] == 0:
            return None

        # Find the index of the last zero
        zero_indexes = np.nonzero(arr == 0)[0]
        if zero_indexes.size > 0:
            start_index = zero_indexes[-1]
        else:
            return None

        # Use np.diff to find where the array stops increasing
        diffs = np.diff(arr[start_index:])

        # Find where the difference is less than or equal to zero (non-increasing sequence)
        non_increasing = np.where(diffs == 0)[0]

        if non_increasing.size > 0:
            # The end index is the last non-increasing index + 1 to account for the difference in np.diff indexing
            end_index = non_increasing[0] + start_index
        else:
            end_index = (
                arr.size - 1
            )  # If the array always increases, end at the last index

        return start_index, end_index

    def _linear_interpolate_nan(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate NaN values in a 1D array.
        Ignores when the slope is negative.
        """
        # fill trailing and ending NaNs with 0
        start_index = np.argmax(~np.isnan(y))
        y[:start_index] = 0
        end_index = len(y) - np.argmax(~np.isnan(y[::-1]))
        y[end_index:] = 0
        # Ensure input arrays are numpy arrays
        nan_mask = np.isnan(y)
        xp = x[~nan_mask]
        x = x[nan_mask]
        yp = y[~nan_mask]

        # Find indices where the right side of the interval for each x would be
        idx = np.searchsorted(xp, x) - 1
        idx[idx < 0] = 0
        idx[idx >= len(xp) - 1] = len(xp) - 2

        # Compute the slope (dy/dx) between the interval points
        slope = (yp[idx + 1] - yp[idx]) / (xp[idx + 1] - xp[idx])
        positive_slope_mask = slope > 0

        # Create a combined mask for NaN positions with positive slopes
        combined_mask = np.zeros_like(y, dtype=bool)
        combined_mask[nan_mask] = positive_slope_mask

        # Compute the interpolated values
        interpolated_values = (yp[idx] + slope * (x - xp[idx]))[positive_slope_mask]
        y[combined_mask] = interpolated_values

        # convert nan to zero
        return np.nan_to_num(y)
