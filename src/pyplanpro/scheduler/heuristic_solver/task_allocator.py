from typing import List, Optional, Tuple, Union

import numpy as np


class TaskAllocator:
    def create_matrix(self, resource_windows_dict) -> np.ndarray:
        """
        Creates a matrix representation of the resource windows. The matrix is used to calculate
        the optimal task allocation across the given resource windows.
        """
        boundaries = np.unique(
            [
                interval
                for windows in resource_windows_dict.values()
                for window in windows
                for interval in window
            ]
        )
        matrix = self._pad_array_with_zeros(boundaries, len(resource_windows_dict))
        for col_index, windows in enumerate(resource_windows_dict.values()):
            for i, window in enumerate(windows):
                self._fill_window(matrix, window, col_index + 1)
        matrix[:, 1:] = np.apply_along_axis(
            self._cumsum_reset_at_minus_one, 0, matrix[:, 1:]
        )
        return matrix

    def find_earliest_solution(
        self, task_duration, resource_windows_dict, resource_count=1
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Finds the earliest possible solution for a given task based on its duration and the
        number of resources available. The method uses a matrix representation of the resource
        windows to calculate the optimal allocation of the task.
        """
        matrix = self.create_matrix(resource_windows_dict)
        resource_matrix = matrix[:, 1:]
        if resource_count == 1 and task_duration > resource_matrix.max():
            print("no soluton found")
            return None

        masked_resource_matrix = self._mask_smallest_except_k_largest(
            resource_matrix, resource_count
        )
        arr_sum = np.sum(masked_resource_matrix, axis=1)
        if task_duration > arr_sum.max():
            print("no soluton found")
            return None
        solution_index = np.argmax(arr_sum >= task_duration)
        solution_resource_cols = ~masked_resource_matrix.mask[solution_index]
        resources = [
            k for k, v in zip(resource_windows_dict, solution_resource_cols) if v
        ]
        solution_matrix = self._linear_expand_matrix(
            matrix[: solution_index + 1, np.insert(solution_resource_cols, 0, True)],
            solution_index,
        )
        allocated_windows = self._solution_to_resource_windows(
            solution_matrix, task_duration, resources
        )
        return allocated_windows

    def _solution_to_resource_windows(
        self, solution_matrix, task_duration, resources
    ) -> List[Tuple[float, float]]:
        """
        Transforms the solution matrix into resource windows that indicate where the task will
        be allocated for each resource.
        """
        resource_matrix = solution_matrix[:, 1:]
        resource_count = resource_matrix.shape[1]

        intervals = solution_matrix[:, 0]
        arr = resource_matrix.sum(axis=1)

        solution_index = np.argmax(arr >= task_duration)
        resource_indexes = self._get_resource_start_end_indexes(
            resource_matrix, solution_index
        )
        print(resource_indexes, resource_matrix)

        resource_windows_dict = {
            resource_id: (intervals[start_index], intervals[end_index])
            for resource_id, (start_index, end_index) in zip(
                resources, resource_indexes
            )
            if start_index != end_index
        }
        return resource_windows_dict

    def _window_start_index(self, arr):
        zero_indices = np.nonzero(arr == 0)  # Find the indices of zeros from the end

        if zero_indices[0].size > 0:
            return zero_indices[0][-1]
        else:
            return 0

    def _window_end_index(self, arr, start_index=0):
        diff = np.diff(arr[start_index:])
        indices = np.where(diff <= 0)[0]
        if indices.size > 0:
            return indices[0] + start_index
        else:
            return len(arr) - 1

    def _get_resource_start_end_indexes(self, resource_matrix, end_index):
        resource_indexes = []
        for arr in resource_matrix.T:
            start_index = self._window_start_index(arr)
            end_index = self._window_end_index(arr, start_index)
            resource_indexes.append((start_index, end_index))
        return resource_indexes

    def _pad_array_with_zeros(self, arr, num_cols) -> np.ndarray:
        """
        Pads a given array with zeros. This is a helper method used in the creation of the
        resource windows matrix.
        """
        new_arr = np.zeros((arr.shape[0], num_cols + 1))
        new_arr[:, 0] = arr
        return new_arr

    def _distribute(self, total, timepoints) -> np.ndarray:
        """
        Distributes a total value across an array of timepoints. This is a helper method used
        in the filling of the resource windows matrix.
        """
        timepoints = np.array(timepoints) - np.min(timepoints)
        diffs = np.diff(np.concatenate([[0], timepoints]))
        ratio_array = diffs / np.sum(diffs)
        return np.round(ratio_array * total).astype(int)

    def _fill_window(self, matrix, window, column_index) -> np.ndarray:
        """
        Fills a specific window of a matrix with distributed values. This is a helper method
        used in the creation of the resource windows matrix.
        """
        first_start_index = np.searchsorted(matrix[:, 0], window[0][0])
        matrix[first_start_index, column_index] = -1
        for start, stop in window:
            index_start = np.searchsorted(matrix[:, 0], start) + 1
            index_stop = np.searchsorted(matrix[:, 0], stop) + 1
            total_duration = stop - start
            timepoints = matrix[index_start - 1 : index_stop, 0]
            distributed = self._distribute(total_duration, timepoints)
            matrix[index_start:index_stop, column_index] = distributed[1:]
        return matrix

    def _cumsum_reset_at_minus_one(self, a) -> np.ndarray:
        """
        Computes the cumulative sum of an array but resets the sum to zero whenever a -1 is encountered.
        This is a helper method used in the creation of the resource windows matrix.
        """
        reset_at = a == -1
        a[a == -1] = 0
        without_reset = a.cumsum()
        overcount = np.maximum.accumulate(without_reset * reset_at)
        return without_reset - overcount

    def _mask_smallest_except_k_largest(self, array, k) -> np.ma.core.MaskedArray:
        """
        Masks the smallest elements in an array, except for the k largest elements on each row.
        This is a helper method used in the finding of the earliest solution.
        """
        indices = np.argpartition(array, -k, axis=1)
        mask = np.ones_like(array, dtype=bool)
        rows = np.arange(array.shape[0])[:, np.newaxis]
        mask[rows, indices[:, -k:]] = False
        masked_array = np.ma.masked_array(array, mask=mask)
        return masked_array

    def _linear_expand_matrix(self, matrix, ending_interval_index) -> np.ndarray:
        """
        Expands the matrix linearly between two intervals. This is a helper method used in the
        finding of the earliest solution.
        """
        end_index = ending_interval_index
        start_index = end_index - 1
        steps = int(matrix[end_index][0] - matrix[start_index][0])
        expaned_interval = np.linspace(
            matrix[start_index], matrix[end_index], steps, False
        )
        return np.concatenate(
            (matrix[:start_index], expaned_interval, matrix[end_index:])
        )
