from collections import namedtuple
from itertools import compress
from typing import Optional

import numpy as np

from factryengine.models import Assignment

Matrix = namedtuple("Matrix", ["resource_ids", "matrix"])


class TaskAllocator:
    def allocate_task(
        self,
        resource_windows_dict: dict[int, np.array],
        assignments: list[Assignment],
        task_duration: float,
    ) -> Optional[dict[int, np.array]]:
        resource_matrix = self.create_resource_coordinates_matrix(resource_windows_dict)
        assignment_matrix = self.create_assignment_matrix(
            assignments[0], resource_matrix
        )

        solution_matrix, solution_resource_ids = self.solve_matrix(
            matrix=assignment_matrix,
            task_duration=task_duration,
        )
        if solution_matrix is None:
            return None

        # get allocated windows
        allocated_windows = self._get_resource_intervals(
            solution_matrix, solution_resource_ids
        )

        return allocated_windows

    def create_matrix(self, windows: list[np.array]) -> np.array:
        intervals_flattened = np.concatenate(
            [np.concatenate(window.tolist()) for window in windows]
        ).flatten()

        boundaries = np.unique(intervals_flattened)

        matrix = [boundaries]
        for window in windows:
            window = self._transform_array(window)
            window[:, 1] = self._cumsum_reset_at_minus_one(window[:, 1])
            new_boundaries = np.setdiff1d(boundaries, window[:, 0])
            window = self._expand_array(window, new_boundaries)
            matrix.append(window[:, 1])

        return np.stack(matrix, axis=-1)

    def solve_matrix(
        self,
        matrix: Matrix,
        task_duration: float,
        resource_count=1,
    ) -> Optional[dict[int, np.array]]:
        """
        Finds the earliest possible solution for a given task based on its duration and
        the number of resources available. The method uses a matrix representation of
        the resource windows to calculate the optimal allocation of the task.
        """
        resource_ids = matrix.resource_ids
        matrix = matrix.matrix

        resource_matrix = matrix[:, 1:]
        if resource_count == 1 and task_duration > resource_matrix.max():
            return (None, None)

        # mask all but the largest group per row if there are multiple groups
        # if len(resource_group_indices) > 1:
        #     resource_matrix = self._fill_array_except_largest_group_per_row(
        #         resource_matrix, resource_group_indices
        #     )

        # mask all but the k largest elements per row
        masked_resource_matrix = self._mask_smallest_except_k_largest(
            resource_matrix, resource_count
        )

        arr_sum = np.sum(masked_resource_matrix, axis=1)
        if task_duration > arr_sum.max():
            return (None, None)

        # get solution index and resource ids
        solution_index = np.argmax(arr_sum >= task_duration)
        solution_resources_mask = ~masked_resource_matrix.mask[solution_index]
        solution_resource_ids = list(compress(resource_ids, solution_resources_mask))

        # solve matrix
        solution_cols_mask = np.concatenate([[True], solution_resources_mask])
        solution_matrix = matrix[: solution_index + 1, solution_cols_mask]
        solution = self._solve_task_end(solution_matrix[-2:], task_duration)
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

    def _get_window_start_index(self, arr):
        """
        returns the index of the first non-zero element in an array from the end.
        """
        zero_indices = np.nonzero(arr == 0)  # Find the indices of zeros from the end

        if zero_indices[0].size > 0:
            return zero_indices[0][-1]
        else:
            return 0

    def _get_resource_intervals(
        self, solution_matrix: np.array, resource_ids: list[tuple[int]]
    ) -> dict[int, list[tuple[float, float]]]:
        """
        gets the resource intervals from the solution matrix.
        """
        start_indexes = [
            self._get_window_start_index(resource_arr)
            for resource_arr in solution_matrix[:, 1:].T
        ]
        end_index = solution_matrix.shape[0] - 1

        resource_windows_dict = {}

        for i, (group_resource_ids, start_index) in enumerate(
            zip(resource_ids, start_indexes)
        ):
            if start_index < end_index:
                for resource_id in group_resource_ids:
                    resource_windows_dict[resource_id] = self._split_intervals(
                        solution_matrix[start_index:, [0, i + 1]]
                    )

        return resource_windows_dict

    def _split_intervals(self, arr):
        """
        splits an array into intervals based on the values in the second column.
        Splitting is done when the value in the second column does not change.
        """
        diff = np.diff(arr[:, 1])
        indices = np.where(diff == 0)[0]
        splits = np.split(arr[:, 0], indices + 1)
        intervals = [
            (round(np.min(split), 2), round(np.max(split), 2))
            for split in splits
            if split.size > 1
        ]
        return intervals

    def _mask_smallest_except_k_largest(self, array, k) -> np.ma.core.MaskedArray:
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

    def _fill_array_except_largest_group_per_row(
        self, array, group_indices=list[list[int, int]]
    ) -> np.array:
        """
        Returns an array where all elements in each row are filled with zero except
        those in the group (set of columns) with the largest sum. Groups are defined by
        a list of lists, each inner list containing the indices of columns in that
        group. Originally zero elements and those not in the largest sum group are
        filled withzeros.
        """
        num_rows = array.shape[0]
        # Initialize mask with all True (masked)
        mask = np.ones_like(array, dtype=bool)

        # Iterate over each row in the array
        for i in range(num_rows):
            row = array[i]
            # Calculate the sums for each group in the row
            group_sums = [np.sum(row[group]) for group in group_indices]
            # Find the indices of the group with the largest sum
            largest_group = group_indices[np.argmax(group_sums)]
            # Unmask (False) the elements in the largest group
            mask[i, largest_group] = False

        # Ensure all zeros in the array are masked
        mask[array == 0] = True

        # Apply the mask to the array
        masked_array = np.ma.masked_array(array, mask=mask)

        # Fill masked values with zeros
        filled_array = masked_array.filled(0)

        return filled_array

    def _cumsum_reset_at_minus_one(self, a) -> np.ndarray:
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

    def _transform_array(self, window):
        # Separate the start/end values and start/end duration values
        interval_values = np.concatenate(window[["start", "end"]].tolist())
        durations = np.concatenate(window[["is_split", "duration"]].tolist())

        # Stack the interval values and durations together
        result = np.column_stack((interval_values, durations))

        return result

    def _expand_array(self, arr: np.ndarray, new_boundaries: np.ndarray) -> np.ndarray:
        """
        Expands an array with new boundaries and calculates the corresponding durations
        using linear interpolation.
        """
        new_boundaries = np.asarray(new_boundaries)

        # Only keep the boundaries that are not already in the array and within the
        # existing range
        mask = (
            ~np.isin(new_boundaries, arr[:, 0])
            & (new_boundaries >= arr[0, 0])
            & (new_boundaries <= arr[-1, 0])
        )
        filtered_boundaries = new_boundaries[mask]

        # Find the indices where the new boundaries fit
        idxs = np.searchsorted(arr[:, 0], filtered_boundaries)

        # Calculate the weights for linear interpolation
        weights = (filtered_boundaries - arr[idxs - 1, 0]) / (
            arr[idxs, 0] - arr[idxs - 1, 0]
        )

        duration_increase = weights * (arr[idxs, 1] - arr[idxs - 1, 1])

        # ensure duration cannot decrease
        duration_increase[duration_increase < 0] = 0

        # Calculate the new durations using linear interpolation
        new_durations = arr[idxs - 1, 1] + duration_increase

        # Combine the new boundaries and durations into an array
        new_rows_within_range = np.column_stack((filtered_boundaries, new_durations))

        # Handle boundaries that are outside the existing range
        new_rows_outside_range = np.column_stack(
            (new_boundaries[~mask], np.zeros(np.sum(~mask)))
        )

        # Combine the old boundaries/durations and new boundaries/durations and sort by
        # the boundaries
        combined = np.vstack((arr, new_rows_within_range, new_rows_outside_range))
        combined = combined[np.argsort(combined[:, 0])]

        return combined

    def interpolate_y_values(self, array):
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

    def replace_missing_boundaries_with_nan(self, array, mask):
        result_array = np.full(mask.shape, np.nan)  # Initialize with nan
        result_array[mask] = array
        return result_array

    def create_resource_coordinates_matrix(self, windows_dict):
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

            duration_colunm = self.replace_missing_boundaries_with_nan(
                window_durations_cumsum, missing_boundaries_mask
            )

            columns.append(duration_colunm)

        coordinates = np.dstack((columns))
        coordinates_nan_filled = self.interpolate_y_values(coordinates)
        return Matrix(
            resource_ids=list(windows_dict.keys()), matrix=coordinates_nan_filled[0]
        )

    def create_assignment_matrix(
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
            # TODO raise error
            pass

        return Matrix(resource_ids=resource_ids, matrix=output_matrix)
