from typing import Optional

import numpy as np


class TaskAllocator:
    def allocate_task(
        self,
        resource_windows: list[np.array],
        resource_ids: np.array,
        task_duration: float,
        resource_count: int,
        resource_group_indices: list[list[int]],
    ) -> Optional[dict[int, np.array]]:
        matrix = self.create_matrix(resource_windows)

        solution_matrix, solution_resource_ids = self.solve_matrix(
            matrix=matrix,
            task_duration=task_duration,
            resource_ids=resource_ids,
            resource_count=resource_count,
            resource_group_indices=resource_group_indices,
        )
        if solution_matrix is None:
            return None
        # get allocated windows
        allocated_windows = self._get_resource_intervals(
            solution_matrix, solution_resource_ids
        )

        return allocated_windows

    def create_matrix(self, windows: list[np.array]) -> np.array:
        boundaries = np.unique(
            np.concatenate([window[:, 0:2].flatten() for window in windows])
        )
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
        matrix: np.array,
        task_duration: float,
        resource_ids: np.array,
        resource_count=1,
        resource_group_indices=list[list[int]],
    ) -> Optional[dict[int, np.array]]:
        """
        Finds the earliest possible solution for a given task based on its duration and
        the number of resources available. The method uses a matrix representation of
        the resource windows to calculate the optimal allocation of the task.
        """

        resource_matrix = matrix[:, 1:]
        if resource_count == 1 and task_duration > resource_matrix.max():
            return (None, None)

        # mask all but the largest group per row if there are multiple groups
        if len(resource_group_indices) > 1:
            resource_matrix = self._fill_array_except_largest_group_per_row(
                resource_matrix, resource_group_indices
            )

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
        solution_resource_ids = resource_ids[solution_resources_mask]

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

    def _get_resource_intervals(self, solution_matrix, resources):
        """
        gets the resource intervals from the solution matrix.
        """
        start_indexes = [
            self._get_window_start_index(resource_arr)
            for resource_arr in solution_matrix[:, 1:].T
        ]
        end_index = solution_matrix.shape[0] - 1

        resource_windows_dict = {
            resource_id: (
                self._split_intervals(solution_matrix[start_index:, [0, i + 1]])
            )
            for i, (resource_id, start_index) in enumerate(
                zip(resources, start_indexes)
            )
            if start_index < end_index
        }
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

    def _transform_array(self, arr):
        # Separate the start/end values and start/end duration values
        interval_values = arr[:, [0, 1]].ravel()
        durations = arr[:, [3, 2]].ravel()

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
