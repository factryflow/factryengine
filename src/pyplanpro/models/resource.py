import numpy as np
from pydantic import BaseModel


class Resource(BaseModel):
    id: int
    available_windows: list[tuple[int, int]] = []
    efficiency_multiplier: float = 1

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Resource):
            return self.id == other.id
        return False

    def merge_intervals(self) -> None:
        """
        Merges overlapping intervals in available_windows.

        The method sorts the available windows based on the start times,
        and then merges overlapping intervals.
        """
        if len(self.available_windows) == 0:
            return

        # Convert the available_windows to a numpy array
        windows = np.array(self.available_windows, dtype=np.int32)

        # Sort the array by start times
        windows = windows[np.argsort(windows[:, 0])]

        # Find where the current start is greater than the previous end
        idx = np.where(windows[:-1, 1] < windows[1:, 0])[0]

        # Stack end indices for intervals to merge with the next start indices
        end_indices = np.hstack((idx, len(windows) - 1))
        start_indices = np.hstack((0, idx + 1))

        # The following line of code creates the merged intervals
        self.available_windows = [
            (windows[i, 0], windows[j, 1]) for i, j in zip(start_indices, end_indices)
        ]
