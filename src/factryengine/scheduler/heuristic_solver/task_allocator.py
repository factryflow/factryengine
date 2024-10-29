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

        if assignments_matrix and constraints_matrix:
            # find the solution for assignments
            solution_matrix = self._solve_matrix(
                matrix=assignments_matrix,
                task_duration=task_duration,
            )

            # find the solution for constraints
            constraints_solution = self._solve_matrix(
                matrix=constraints_matrix,
                task_duration=task_duration,
            )
        else:
            # matrix to solve
            matrix_to_solve = assignments_matrix or constraints_matrix

            # find the solution
            solution_matrix = self._solve_matrix(
                matrix=matrix_to_solve,
                task_duration=task_duration,
            )

        # process solution to find allocated resource windows
        allocated_windows = self._get_resource_intervals(
            matrix=solution_matrix
        )

        # add constraints to allocated windows
        if constraints and assignments:
            constraints_matrix_trimmed = Matrix.trim_end(
                original_matrix=constraints_solution, trim_matrix=solution_matrix
            )
            allocated_windows.update(
                self._get_resource_intervals(
                    matrix=constraints_matrix_trimmed
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
        self, matrix: Matrix
    ) -> dict[int, list[tuple[int, int]]]:
        """
        Extracts all the resource intervals used from the solution matrix, 
        including non-contiguous intervals and partial usage.
        """
        resource_windows_output = {}

        # Iterate over each resource and its corresponding matrix intervals
        for resource_id, resource_intervals in zip(matrix.resource_ids, matrix.resource_matrix.T):
            

            # Get all relevant indexes
            indexes = self._find_indexes(resource_intervals)

            # Pair the indexes in groups of 2 (start, end)
            intervals = []
            for start, end in zip(indexes[::2], indexes[1::2]):
                # Use start and end indexes directly without skipping
                # print(f"Start: {start}, End: {end}")
                interval_start = matrix.intervals[start]
                interval_end = matrix.intervals[end]

                # Append the interval to the list
                intervals.append((int(np.round(interval_start)), int(np.round(interval_end))))

            # Store the intervals for the current resource
            resource_windows_output[resource_id] = intervals

        return resource_windows_output
    
    def _find_first_index(self, resource_intervals: np.ma.MaskedArray) -> int | None:
        # Shift the mask by 1 to align with the 'next' element comparison
        current_mask = resource_intervals.mask[:-1]
        next_mask = resource_intervals.mask[1:]
        next_values = resource_intervals.data[1:]

        # Vectorized condition: current is masked, next is non-masked, and next value > 0
        condition = (current_mask) & (~next_mask) & (next_values > 0)

        # Find the first index where the condition is met
        indices = np.where(condition)[0]

        first_index = indices[0] if len(indices) > 0 else 0

        next_non_zero_index = np.where(
            (~resource_intervals.mask[first_index + 2:])  # Non-masked (non-zero)
                & (resource_intervals.mask[first_index + 1:-1])  # Previous value masked
        )[0]

        # Adjust x to align with the original array's indices
        next_non_zero_index = (
            (first_index + 2 + next_non_zero_index[0]) if len(next_non_zero_index) > 0 else None
        )

        if next_non_zero_index and resource_intervals[next_non_zero_index] == resource_intervals[first_index+1]:
            first_index = next_non_zero_index

        return first_index


    def _find_indexes(self, resource_intervals: np.ma.MaskedArray) -> int | None:
        """
        Finds relevant indexes in the resource intervals where the resource is used.
        """
        # Mask where the data in resource_intervals is 0
        resource_intervals = np.ma.masked_where(resource_intervals == 0.0, resource_intervals)

        indexes = []
        first_index = self._find_first_index(resource_intervals)
        last_index = resource_intervals.size-1

        indexes = [first_index]  # Start with the first index
        is_last_window_start = True  # Flag to indicate the start of a window

        # Iterate through the range between first and last index
        for i in range(first_index + 1, last_index + 1):
            current = resource_intervals[i]
            previous = resource_intervals[i - 1] if i > 0 else 0
            next_value = resource_intervals[i + 1] if i < last_index else 0

            # Check if the current value is masked
            is_prev_masked = resource_intervals.mask[i - 1] if i > 0 else False
            is_curr_masked = resource_intervals.mask[i]
            is_next_masked = resource_intervals.mask[i+1] if i < last_index else False

            # Skip if all values are the same (stable window)
            if current > 0 and current == previous == next_value:
                continue

            # Skip increasing trend from masked value
            if current > 0 and current < next_value and is_prev_masked:
                continue

            # Detect end of a window
            if current > 0 and current == next_value and (is_prev_masked or previous < current) and is_last_window_start:
                indexes.append(i)
                is_last_window_start = False
                continue

            # Detect end of window using masks 
            if current > 0 and is_next_masked and not is_curr_masked and is_last_window_start:
                indexes.append(i)
                is_last_window_start = False
                continue

            # Detect start of a new window
            if current > 0 and next_value > current and (is_prev_masked or previous == current) and not is_last_window_start:
                indexes.append(i)
                is_last_window_start = True
                continue

            # Detect start of window using masks
            if is_curr_masked and next_value > 0 and (previous > 0 or is_prev_masked) and not is_last_window_start:
                indexes.append(i)
                is_last_window_start = True
                continue

            # Always add the last index
            if i == last_index:
                indexes.append(i)

        # Return the first valid index, or None if no valid index is found
        return indexes

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
        Computes the cumulative sum but resets to 0 whenever a -1 is encountered.
        """
        reset_mask = (a == -1)
        a[reset_mask] = 0  # Replace -1 with 0 for sum calculation
        cumsum_result = np.cumsum(a)
        cumsum_result[reset_mask] = 0  # Reset at gaps

        return cumsum_result

    def _cumsum_reset_at_minus_one_2d(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies cumulative sum along the columns of a 2D array and resets at gaps (-1).
        """
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

            # replace -1 with 0
            is_split_mask = resource_column == -1
            resource_column[is_split_mask] = 0

            # fill nan values with linear interpolation
            resource_column = self._linear_interpolate_nan(resource_column, intervals)

            # distribute the window durations over the intervals
            resource_column = self._diff_and_zero_negatives(resource_column)

            # restore -1
            resource_column[is_split_mask] = -1

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

    def _diff_and_zero_negatives(self, arr: np.ndarray) -> np.ndarray:
        """
        subtracts the previous element from each element in an array and replaces negative.
        """
        # compute the difference
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


    # def _find_indexes(self, arr: np.array) -> tuple[int, int] | None:
    #     """
    #     Find the start and end indexes for a valid segment of resource availability.
    #     This version avoids explicit loops and ensures the start index is correctly identified.
    #     """
    #     # If the input is a MaskedArray, handle it accordingly
    #     if isinstance(arr, np.ma.MaskedArray):
    #         arr_data = arr.data
    #         mask = arr.mask
    #         # Find valid (unmasked and positive) indices
    #         valid_indices = np.where((~mask) & (arr_data >= 0))[0]
    #     else:
    #         valid_indices = np.where(arr >= 0)[0]

    #     # If no valid indices are found, return None (no available resources)
    #     if valid_indices.size == 0:
    #         return None

    #     # Identify if the start of the array is valid
    #     start_index = 0 if arr[0] > 0 else valid_indices[0]

    #     # Calculate differences between consecutive indices
    #     diffs = np.diff(valid_indices)

    #     # Identify segment boundaries where there is a gap greater than 1
    #     gaps = diffs > 1
    #     segment_boundaries = np.where(gaps)[0]

    #     # Insert the start index explicitly to ensure it is considered
    #     segment_starts = np.insert(segment_boundaries + 1, 0, 0)
    #     segment_ends = np.append(segment_starts[1:], len(valid_indices))

    #     # Always take the first segment (which starts at the earliest valid index)
    #     start_pos = segment_starts[0]
    #     end_pos = segment_ends[0] - 1

    #     # Convert these segment positions to the actual start and end indices
    #     start_index = valid_indices[start_pos]
    #     end_index = valid_indices[end_pos]

    #     return start_index, end_index



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
