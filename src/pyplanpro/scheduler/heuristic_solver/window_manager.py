import numpy as np

from ...models.resource import Resource


class WindowManager:
    def __init__(self, resources: list[Resource]):
        self.resources = resources
        self.resource_windows_dict = self.create_resource_windows_dict()

    def create_resource_windows_dict(self) -> dict[int, np.ndarray]:
        """
        Creates a dictionary mapping resource IDs to numpy arrays representing windows.
        """
        return {
            resource.id: self.windows_to_numpy(resource.available_windows)
            for resource in self.resources
        }

    def windows_to_numpy(self, windows: list[tuple[int, int]]) -> np.ndarray:
        """
        Converts a list of windows to a numpy array.
        """
        arr = np.array(windows)
        return np.concatenate([arr, np.diff(arr), np.zeros((arr.shape[0], 1))], axis=1)

    def trim_windows(
        self, windows: np.ndarray, trim_interval: tuple[int, int]
    ) -> np.ndarray:
        """
        Trims the provided windows based on the provided trim window.
        """
        trim_start, trim_end = trim_interval

        # Find the range of intervals that could potentially overlap with trim_interval
        start_idx = np.searchsorted(windows[:, 1], trim_start, side="right")
        end_idx = np.searchsorted(windows[:, 0], trim_end, side="left")

        # If no overlap, return original intervals
        if start_idx == end_idx:
            return windows

        # Identify intervals for trimming or removing
        overlap_windows = windows[start_idx:end_idx]
        mask_end = overlap_windows[:, 1] <= trim_end
        mask_start = overlap_windows[:, 0] >= trim_start
        mask_delete = np.logical_and(mask_end, mask_start)

        print(overlap_windows)
        # Trim the end of intervals that start before and end within the trim_interval
        if mask_end.any():
            overlap_windows[mask_end, 1] = trim_start
            overlap_windows[mask_end, 2] = (
                overlap_windows[mask_end, 1] - overlap_windows[mask_end, 0]
            )
            end_idx_temp = min(end_idx, windows.shape[0] - 1)  # handle out of bounds
            windows[end_idx_temp, 3] = 1

        # Trim the start of intervals that start within and end after the trim_interval
        if mask_start.any():
            overlap_windows[mask_start, 0] = trim_end
            overlap_windows[mask_start, 2] = (
                overlap_windows[mask_start, 1] - overlap_windows[mask_start, 0]
            )
            overlap_windows[mask_start, 3] = 1

        # Replace the old intervals with the updated ones, and delete fully overlapped ones  # noqa: E501
        windows[start_idx:end_idx] = overlap_windows

        if mask_delete.any():
            windows = np.delete(
                windows, np.arange(start_idx, end_idx)[mask_delete], axis=0
            )

        return windows
