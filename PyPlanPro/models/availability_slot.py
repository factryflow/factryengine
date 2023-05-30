from intervaltree import IntervalTree, Interval

class AvailabilitySlot():

    def __init__(self, intervals):
        self.tree = IntervalTree.from_tuples(intervals)

    def intersection(self, other):
        result = IntervalTree()
        for interval in self.tree:
            overlapping = other.overlap(interval.begin, interval.end)
            for overlap in overlapping:
                result.add(Interval(max(interval.begin, overlap.begin), min(interval.end, overlap.end)))
        result.merge_overlaps(strict=False)
        return result
    
    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else float('inf')
            result_intervals = []
            for interval in self.tree:
                if interval.begin < stop and interval.end > start:
                    result_intervals.append((max(start, interval.begin), min(stop, interval.end)))
            return AvailabilitySlot(result_intervals)
        else:
            raise TypeError("Indices must be slices")
    
    def __repr__(self):
            return self.tree.__repr__().replace("IntervalTree","AvailabilitySlot")
    
    def get_duration(self):
        return sum(interval.end - interval.begin for interval in self.tree)
    
    def start(self):
        return self.tree.begin()
    
    def end(self):
        return self.tree.end()
