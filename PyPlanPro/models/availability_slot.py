from intervaltree import IntervalTree, Interval
import numpy as np

class UnscheduledQueue():
    def __init__(self, windows = None):
        self.windows = sorted(windows) if windows else []

    def add_window(self, window):
        self.windows.append(window)
        self.windows.sort()  # Sort by start time

    def remove_window(self, window):
        self.windows.remove(window)

    def _get_allocateable_window(self, task_duration, earliest_start):
        for window in self.windows:
            if window[earliest_start:].duration >= task_duration:
                return window

    def get_earliest_task_window(self, task_duration, earliest_start=0):
        # find allocateable window
        allocateable_window = self._get_allocateable_window(task_duration, earliest_start)
        if allocateable_window is None:
            return None
        
        # find task window
        remaining_duration = task_duration
        task_start = allocateable_window.begin()
        
        for interval in sorted(allocateable_window.tree):
            start, end = interval.begin, interval.end
            interval_duration = end-start
            
            if interval_duration >= remaining_duration:
                task_end = end - (interval_duration - remaining_duration)
                task_window = allocateable_window[task_start: task_end]
                return task_window 
            
            remaining_duration -= interval_duration

    def __repr__(self):
        return f"UnscheduledQueue({self.windows})"

class UnscheduledWindow():

    def __init__(self, intervals):
        self.arr = self.numpy_array_from_tuples(intervals)
        self._duration = None

    def numpy_array_from_tuples(self, intervals):
        arr = np.concatenate([np.arange(start, end+1) for start, end in intervals])
        arr.sort()
        return np.unique(arr)
            

    # def intersection(self, other):
    #     result = IntervalTree()
    #     for interval in self.tree:
    #         overlapping = other.overlap(interval.begin, interval.end)
    #         for overlap in overlapping:
    #             result.add(Interval(max(interval.begin, overlap.begin), min(interval.end, overlap.end)))
    #     result.merge_overlaps(strict=False)
    #     return result
    
    # def __getitem__(self, item):
    #     if isinstance(item, slice):
    #         start = item.start if item.start is not None else 0
    #         stop = item.stop if item.stop is not None else float('inf')
    #         result_intervals = []
    #         for interval in self.tree:
    #             if interval.begin < stop and interval.end > start:
    #                 result_intervals.append((max(start, interval.begin), min(stop, interval.end)))
    #         return UnscheduledWindow(result_intervals)
    #     else:
    #         raise TypeError("Indices must be slices")
    
    # def __repr__(self):
    #         return self.tree.__repr__().replace("IntervalTree","UnscheduledWindow")
    
    # def get_duration(self):
    #     return sum(interval.end - interval.begin for interval in self.tree)
    
    # @property
    # def duration(self):
    #     if self._duration is None:  # Calculate duration if not cached
    #         self._duration = sum(interval.end - interval.begin for interval in self.tree)
    #     return self._duration
    
    # def begin(self):
    #     return self.tree.begin()
    
    # def end(self):
    #     return self.tree.end()
    
    # def __lt__(self, other):
    #     return self.tree.begin() < other.tree.begin()
