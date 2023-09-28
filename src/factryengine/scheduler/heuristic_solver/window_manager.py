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

    def get_task_resource_windows(
        self, task_resource_ids: list[int], task_earliest_start: int
    ) -> list[np.ndarray]:
        """
        Returns the resource windows for the resource ids.
        The windows are trimmed to the min_task_start.
        """
        return [
            trimmed_window
            for resource_id in task_resource_ids
            if (
                trimmed_window := self.trim_window(
                    window=self.resource_windows_dict[resource_id],
                    trim_interval=(0, task_earliest_start),
                )
            ).size
            > 0
        ]

    def windows_to_numpy(self, windows: list[tuple[int, int]]) -> np.ndarray:
        """
        Converts a list of windows to a numpy array.
        """
        arr = np.array(windows)
        return np.concatenate([arr, np.diff(arr), np.zeros((arr.shape[0], 1))], axis=1)

    def update_resource_windows(
        self, allocated_resource_windows_dict: dict[int, list[tuple[int, int]]]
    ) -> None:
        """
        Removes the task interaval from the resource windows
        """
        for resource_id, trim_interval in allocated_resource_windows_dict.items():
            window = self.resource_windows_dict[resource_id]
            self.resource_windows_dict[resource_id] = self.trim_window(
                window, trim_interval
            )

    def trim_window(
        self, window: np.ndarray, trim_interval: tuple[int, int]
    ) -> np.ndarray:
        """
        Trims the provided windows based on the provided trim window.
        """
        window = window.copy()
        trim_start, trim_end = trim_interval

        start_idx = np.searchsorted(window[:, 1], trim_start, side="right")
        end_idx = np.searchsorted(window[:, 0], trim_end, side="left")

        if start_idx == end_idx:
            return window

        overlap_windows = window[start_idx:end_idx]
        mask_end = overlap_windows[:, 1] <= trim_end
        mask_start = overlap_windows[:, 0] >= trim_start
        mask_delete = np.logical_and(mask_end, mask_start)
        mask_between = np.logical_and(
            overlap_windows[:, 0] < trim_start, overlap_windows[:, 1] > trim_end
        )

        slopes = self._calculate_slopes(
            overlap_windows
        )  # Compute slopes for all overlap_windows

        window = self._handle_mask_between(
            window,
            overlap_windows,
            mask_between,
            slopes,
            trim_start,
            trim_end,
            start_idx,
            end_idx,
        )
        window = self._handle_mask_end(
            window, overlap_windows, mask_end, trim_start, end_idx
        )
        window = self._handle_mask_start(window, overlap_windows, mask_start, trim_end)
        window = self._delete_overlapped_windows(
            window, mask_delete, start_idx, end_idx
        )

        return window

    def _calculate_slopes(self, windows: np.ndarray) -> np.ndarray:
        """
        Calculates the slopes for the given windows.
        """
        return (windows[:, 2] - windows[:, 3]) / (windows[:, 1] - windows[:, 0])

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
            overlap_windows[0, 1] = trim_start
            overlap_windows[0, 2] = (
                overlap_windows[0, 1] - overlap_windows[0, 0]
            ) * slopes_between

            overlap_windows[1, 0] = trim_end
            overlap_windows[1, 2] = (
                overlap_windows[1, 1] - overlap_windows[1, 0]
            ) * slopes_between
            overlap_windows[1, 3] = -1

            return np.concatenate(
                (windows[:start_idx], overlap_windows, windows[end_idx:])
            )

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
            overlap_windows[mask_end, 1] = trim_start
            overlap_windows[mask_end, 2] = (
                overlap_windows[mask_end, 1] - overlap_windows[mask_end, 0]
            )
            end_idx_temp = min(end_idx, windows.shape[0] - 1)  # handle out of bounds
            windows[end_idx_temp, 3] = -1

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
            overlap_windows[mask_start, 0] = trim_end
            overlap_windows[mask_start, 2] = (
                overlap_windows[mask_start, 1] - overlap_windows[mask_start, 0]
            )
            overlap_windows[mask_start, 3] = -1

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
