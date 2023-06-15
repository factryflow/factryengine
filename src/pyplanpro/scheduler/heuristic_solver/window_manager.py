import numpy as np


class WindowManager:
    def __init__(self, resources):
        self.resources = resources

    def create_resource_windows_dict(self):
        return {
            resource.id: self.windows_to_numpy(resource.available_windows)
            for resource in self.resources
        }

    @staticmethod
    def windows_to_numpy(windows):
        arr = np.array(windows)
        return np.concatenate([arr, np.diff(arr), np.zeros((arr.shape[0], 1))], axis=1)

    @staticmethod
    def trim_windows(windows, trim_window):
        # Find the range of intervals that could potentially overlap with trim_interval
        start_idx = np.searchsorted(windows[:, 1], trim_window[0])
        end_idx = np.searchsorted(windows[:, 0], trim_window[1])

        # If no overlap, return original intervals
        if start_idx == end_idx:
            return windows

        # Identify intervals for trimming or removing
        overlap_windows = windows[start_idx:end_idx]
        mask_end = overlap_windows[:, 1] <= trim_window[1]
        mask_start = overlap_windows[:, 0] >= trim_window[0]
        mask_delete = np.logical_and(mask_end, mask_start)

        # Trim the end of intervals that start before and end within the trim_interval
        if mask_end.any():
            overlap_windows[mask_end, 1] = trim_window[0]
            overlap_windows[mask_end, 2] = (
                overlap_windows[mask_end, 1] - overlap_windows[mask_end, 0]
            )
            windows[end_idx, 3] = 1

        # Trim the start of intervals that start within and end after the trim_interval
        if mask_start.any():
            overlap_windows[mask_start, 0] = trim_window[1]
            overlap_windows[mask_start, 2] = (
                overlap_windows[mask_start, 1] - overlap_windows[mask_start, 0]
            )
            overlap_windows[mask_start, 3] = 1

        # Replace the old intervals with the updated ones, and delete fully overlapped ones
        windows[start_idx:end_idx] = overlap_windows

        if mask_delete.any():
            windows = np.delete(
                windows, np.arange(start_idx, end_idx)[mask_delete], axis=0
            )

        return windows
