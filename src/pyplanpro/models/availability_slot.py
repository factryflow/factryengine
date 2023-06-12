from functools import reduce
import numpy as np

class UnscheduledQueue():
    '''takes a list of unscheduled windows'''
    def __init__(self, windows = None):
        self.windows = self.windows_to_numpy(windows) if windows else []

    def windows_to_numpy(self, windows):
        numpy_windows = [self.window_intervals_to_numpy(window) for window in windows]
        
        return sorted(numpy_windows, key=lambda x: x[0])
    
    def window_intervals_to_numpy(self, intervals):
        return np.concatenate([np.arange(start+1, end+1) for start, end in intervals])
    
    def intersection(windows):
        return reduce(np.intersect1d, windows)

    def add_window(self, window):
        self.windows.append(window)
        self.windows.sort()  # Sort by start time

    def remove_window(self, window):
        self.windows.remove(window)

    def _get_allocateable_window(self, task_duration, earliest_start):
        for window in self.windows:
            if window.size < task_duration:
                continue
            if earliest_start in window:
                if window[window >= earliest_start].size < task_duration:
                    continue
                return window[window >= earliest_start]
            return window

    def get_earliest_task_window(self, task_duration, earliest_start=0):
        # find allocateable window
        allocateable_window = self._get_allocateable_window(task_duration, earliest_start)
        print(allocateable_window)
        if allocateable_window is None:
            return None
        
        # find task window
        return allocateable_window[:task_duration]
    
    def get_earliest_task_window(windows, task_duration):
        """
        Finds the quickest completion time of a task. Count is equal to Efficiency at a given minute.
        """
        concatenated_window, count = np.unique(np.concatenate(windows), return_counts = True)
        if count.sum() < task_duration:
            return None
        
        index = np.argmax(count.cumsum() >= task_duration)
        return concatenated_window[:index+1]
    

    def __repr__(self):
        return f"UnscheduledQueue({[(window[0], window[-1]) for window in self.windows for intervals in window]})"
