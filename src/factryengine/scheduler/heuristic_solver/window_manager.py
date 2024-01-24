import numpy as np

from ...models.resource import Resource


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
        Removes the task interaval from the resource windows
        """
        for resource_id, trim_interval in allocated_resource_windows_dict.items():
            window = self.resource_windows_dict[resource_id]
            self.resource_windows_dict[resource_id] = self._trim_window(
                window, trim_interval
            )

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

        # Define the dtype for the structured array
        dtype = [
            ("start", np.float32),
            ("end", np.float32),
            ("duration", np.float32),
            ("is_split", np.int32),
        ]

        # Create an empty structured array with the specified dtype
        result = np.zeros(arr.shape[0], dtype=dtype)

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
            window,
            overlap_intervals,
            mask_between,
            slopes,
            trim_start,
            trim_end,
            start_idx,
            end_idx,
        )
        window = self._handle_mask(
            window, overlap_intervals, mask_end, trim_start, "end", end_idx
        )
        window = self._handle_mask(
            window, overlap_intervals, mask_start, trim_end, "start"
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
            slopes_between = slopes[mask_between]
            overlap_windows = np.concatenate([overlap_windows, overlap_windows])
            overlap_windows[0]["end"] = trim_start
            overlap_windows[0]["duration"] = (
                overlap_windows[0]["end"] - overlap_windows[0]["start"]
            ) * slopes_between

            overlap_windows[1]["end"] = trim_end
            overlap_windows[1]["duration"] = (
                overlap_windows[1]["end"] - overlap_windows[1]["start"]
            ) * slopes_between
            overlap_windows[1]["is_split"] = -1

            return np.concatenate(
                (windows[:start_idx], overlap_windows, windows[end_idx:])
            )

        return windows

    def _handle_mask(
        self,
        windows: np.ndarray,
        overlap_windows: np.ndarray,
        mask: np.ndarray,
        trim_value: int,
        field: str,
        end_idx: int = None,
    ) -> np.ndarray:
        """
        Handles the case where mask is True.
        """
        if np.any(mask):
            overlap_windows[mask][field] = trim_value
            overlap_windows[mask]["duration"] = (
                overlap_windows[mask]["end"] - overlap_windows[mask]["start"]
            )
            if field == "end" and end_idx is not None:
                end_idx_temp = min(
                    end_idx, windows.shape[0] - 1
                )  # handle out of bounds
                windows[end_idx_temp]["is_split"] = -1
            else:
                overlap_windows[mask]["is_split"] = -1

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
