import numpy as np

from ...models.resource import Resource

window_dtype = [
    ("start", np.float32),
    ("end", np.float32),
    ("duration", np.float32),
    ("is_split", np.int32),
]


class WindowManager:
    def __init__(self, resources: list[Resource]):
        self.resources = resources
        self.resource_windows_dict = self._create_resource_windows_dict()

    def get_task_resource_windows_dict(
        self, task_resource_ids: list[int], task_earliest_start: int
    ) -> dict[int, np.ndarray]:
        """
        Returns the resource windows for the resource ids.
        The windows are trimmed to the min_task_start.
        """
        trimmed_windows_dict = {}

        # Loop over each resource ID
        for resource_id in task_resource_ids:
            # Get the window for the current resource ID
            resource_windows = self.resource_windows_dict[resource_id]

            # Trim the window to the task's earliest start time
            trimmed_window = self._trim_window(
                window=resource_windows,
                trim_interval=(0, task_earliest_start),
            )

            # If the trimmed window is not empty, add it to the dictionary
            if trimmed_window.size > 0:
                trimmed_windows_dict[resource_id] = trimmed_window

        return trimmed_windows_dict

    def update_resource_windows(
        self, allocated_resource_windows_dict: dict[int, list[tuple[int, int]]]
    ) -> None:
        """
        Removes the allocated intervals from the resource windows.
        """
        for resource_id, intervals_to_remove in allocated_resource_windows_dict.items():
            if not intervals_to_remove:
                continue

            # Get the resource's current available windows (structured array)
            resource_windows = self.resource_windows_dict[resource_id]

            # Remove each interval individually
            for interval in intervals_to_remove:
                resource_windows = self._remove_interval_from_windows(resource_windows, interval)

            # Update the resource windows dict
            self.resource_windows_dict[resource_id] = resource_windows


    def _remove_interval_from_windows(
        self, windows, interval_to_remove: tuple[int, int]
    ) -> np.ndarray:
        updated_windows = []
        remove_start, remove_end = interval_to_remove

        for window in windows:
            window_start = window['start']
            window_end = window['end']

            # No overlap
            if remove_end <= window_start or remove_start >= window_end:
                updated_windows.append(window)
                continue

            # Interval completely covers the window
            if remove_start <= window_start and remove_end >= window_end:
                continue  # Entire window is removed

            # Overlaps at the start
            if remove_start <= window_start < remove_end < window_end:
                new_window = window.copy()
                new_window['start'] = remove_end
                new_window['duration'] = new_window['end'] - new_window['start']
                updated_windows.append(new_window)
                continue

            # Overlaps at the end
            if window_start < remove_start < window_end <= remove_end:
                new_window = window.copy()
                new_window['end'] = remove_start
                new_window['duration'] = new_window['end'] - new_window['start']
                updated_windows.append(new_window)
                continue

            # Overlaps in the middle, split the window
            if window_start < remove_start and window_end > remove_end:
                # Create two new windows
                window1 = window.copy()
                window1['end'] = remove_start
                window1['duration'] = window1['end'] - window1['start']

                window2 = window.copy()
                window2['start'] = remove_end
                window2['duration'] = window2['end'] - window2['start']

                updated_windows.extend([window1, window2])
                continue

        return np.array(updated_windows, dtype=window_dtype)


    def _create_resource_windows_dict(self) -> dict[int, np.ndarray]:
        """
        Creates a dictionary mapping resource IDs to numpy arrays representing windows.
        """
        return {
            resource.id: self._windows_to_numpy(resource.available_windows)
            for resource in self.resources
        }

    def _windows_to_numpy(self, windows: list[tuple[int, int]]) -> np.ndarray:
        """
        Converts a list of windows to a numpy array.
        """
        # Convert the list of windows to a numpy array
        arr = np.array(windows)

        # Create an empty structured array with the specified dtype
        result = np.zeros(arr.shape[0], dtype=window_dtype)

        # Fill the 'start' and 'end' fields with the first and second columns of 'arr', respectively
        result["start"], result["end"] = arr[:, 0], arr[:, 1]

        # Calculate the duration of each window and fill the 'duration' field
        result["duration"] = np.diff(arr, axis=1).flatten()

        # Fill the 'is_split' field with zeros
        result["is_split"] = 0

        return result

    def _trim_window(
        self, window: np.ndarray, trim_interval: tuple[int, int]
    ) -> np.ndarray:
        """
        Trims the provided windows based on the provided trim window.
        """
        window = window.copy()
        trim_start, trim_end = trim_interval

        start_idx = np.searchsorted(window["end"], trim_start, side="right")
        end_idx = np.searchsorted(window["start"], trim_end, side="left")

        if start_idx == end_idx:
            return window

        overlap_intervals = window[start_idx:end_idx]
        mask_end = overlap_intervals["end"] <= trim_end
        mask_start = overlap_intervals["start"] >= trim_start
        mask_delete = np.logical_and(mask_end, mask_start)
        mask_between = np.logical_and(
            overlap_intervals["start"] < trim_start, overlap_intervals["end"] > trim_end
        )

        slopes = self._calculate_slopes(
            overlap_intervals
        )  # Compute slopes for all overlap_windows

        window = self._handle_mask_between(
            windows=window,
            overlap_windows=overlap_intervals,
            mask_between=mask_between,
            slopes=slopes,
            trim_start=trim_start,
            trim_end=trim_end,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        window = self._handle_mask_start(
            windows=window,
            overlap_windows=overlap_intervals,
            mask_start=mask_start,
            trim_end=trim_end,
        )

        window = self._handle_mask_end(
            windows=window,
            overlap_windows=overlap_intervals,
            mask_end=mask_end,
            trim_start=trim_start,
            end_idx=end_idx,
        )

        window = self._delete_overlapped_windows(
            window, mask_delete, start_idx, end_idx
        )

        return window

    def _calculate_slopes(self, window: np.ndarray) -> np.ndarray:
        """
        Calculates the slopes for the given intervals.
        """
        return window["duration"] / (window["end"] - window["start"])

    def _handle_mask_between(
        self,
        windows: np.ndarray,
        overlap_windows: np.ndarray,
        mask_between: np.ndarray,
        slopes: np.ndarray,
        trim_start: int,
        trim_end: int,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        """
        Handles the case where mask_between is True.
        """
        if np.any(mask_between):
            # Get the slopes between the mask
            slopes_between = slopes[mask_between]

            # Duplicate the overlap windows
            overlap_windows = np.concatenate([overlap_windows, overlap_windows])

            # Update the end and duration of the first overlap window
            overlap_windows[0]["end"] = trim_start
            overlap_windows[0]["duration"] = (
                overlap_windows[0]["end"] - overlap_windows[0]["start"]
            ) * slopes_between

            # Update the end, duration, and is_split of the second overlap window
            overlap_windows[1]["start"] = trim_end
            overlap_windows[1]["duration"] = (
                overlap_windows[1]["end"] - overlap_windows[1]["start"]
            ) * slopes_between
            overlap_windows[1]["is_split"] = -1

            # Concatenate the windows before the start index, the overlap windows, and the windows after the end index
            final_windows = np.concatenate(
                (windows[:start_idx], overlap_windows, windows[end_idx:])
            )

            return final_windows

        return windows

    def _handle_mask_start(
        self,
        windows: np.ndarray,
        overlap_windows: np.ndarray,
        mask_start: np.ndarray,
        trim_end: int,
    ) -> np.ndarray:
        """
        Handles the case where mask_start is True.
        """
        if np.any(mask_start):
            # Update "start" field
            overlap_windows["start"][mask_start] = trim_end

            # Update "duration" field based on updated "start" and existing "end"
            overlap_windows["duration"][mask_start] = (
                overlap_windows["end"][mask_start] - trim_end
            )

            # Update "is_split" field
            overlap_windows["is_split"][mask_start] = -1

        return windows

    def _handle_mask_end(
        self,
        windows: np.ndarray,
        overlap_windows: np.ndarray,
        mask_end: np.ndarray,
        trim_start: int,
        end_idx: int,
    ) -> np.ndarray:
        """
        Handles the case where mask_end is True.
        """
        if np.any(mask_end):
            overlap_windows["end"][mask_end] = trim_start
            overlap_windows["duration"][mask_end] = (
                trim_start - overlap_windows["start"][mask_end]
            )
            end_idx_temp = min(end_idx, windows.shape[0] - 1)  # handle out of bounds
            windows["is_split"][end_idx_temp] = -1

        return windows

    def _delete_overlapped_windows(
        self, windows: np.ndarray, mask_delete: np.ndarray, start_idx: int, end_idx: int
    ) -> np.ndarray:
        """
        Deletes the windows that fully overlap.
        """
        if np.any(mask_delete):
            windows = np.delete(
                windows, np.arange(start_idx, end_idx)[mask_delete], axis=0
            )

        return windows
