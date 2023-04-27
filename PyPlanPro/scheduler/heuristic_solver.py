import networkx as nx

class HeuristicSolver():
    def __init__(self, tasks, resources):
        self.tasks = tasks
        self.resouces = resources

    def solve(self):
        task_ids_ordered = self._get_task_order(self.tasks)
        task_dict = {task.id: task for task in self.tasks}
        task_vars = {task.id: {} for task in self.tasks}
        resource_intervals = self._get_resource_intervals(self.resouces)

        for task_id in task_ids_ordered:
            task = task_dict[task_id]
            # get max_start_time from max predecessors end
            max_start_time = max((task_vars[pred.id].get('task_end',0) for pred in task.predecessors), default=0)
            
            # find the resource who completes the task first
            fastest_resource = self._get_fastest_resource(task,resource_intervals, max_start_time)
            if fastest_resource is None:
                print(f"No resource could be found for task_id = {task.id}")
                continue
            # update task_vars with assigned resource, task_start and task_end
            
            task_values = {
                "assigned_resource": fastest_resource["resource"], 
                "task_start": fastest_resource["start_time"],
                "task_end" : fastest_resource["end_time"]
                }
            
            task_vars[task.id].update(task_values)

        return task_vars
    def _get_task_order(self, tasks):
        """
        Given a dictionary of tasks with durations, predecessors, and priorities, this function returns a list of tasks
        in the order required to complete them as quickly as possible while considering task priorities.

        :return: List of tasks in the order required to complete them as quickly as possible considering task priorities.
        """
        
        # Create a directed acyclic graph
        G = nx.DiGraph()

        # Add nodes and edges to the graph
        for task  in tasks:
            G.add_node(task.id, duration=task.duration, priority=task.priority)
            for predecessor in task.predecessors:
                G.add_edge(predecessor.id, task.id)

        # Compute the longest path for each node using a topological sort
        longest_path = {task.id: 0 for task in tasks}
        for task in nx.topological_sort(G):
            duration = G.nodes[task]['duration']
            for predecessor in G.predecessors(task):
                longest_path[task] = max(longest_path[task], longest_path[predecessor] + duration)

        # Custom topological sort considering priority and longest path
        visited = set()
        result = []

        def visit(node):
            """
            Recursive function to traverse the task graph in the desired order.
            """
            if node not in visited:
                visited.add(node)
                # Sort predecessors by priority and longest path
                predecessors = sorted(
                    G.predecessors(node),
                    key=lambda n: (G.nodes[n]['priority'], -longest_path[n])
                )
                for predecessor in predecessors:
                    visit(predecessor)
                result.append(node)

        # Start visiting tasks, sorting them by priority and longest path
        for task in sorted(tasks, key=lambda t: (G.nodes[t.id]['priority'], -longest_path[t.id])):
            visit(task.id)

        return result

    def _get_fastest_resource(self, task, resource_intervals, max_start_time):
        """Returns the resource with the earliest end time for a task."""
        resource_start_ends = []
        for resource in task.get_resources():
            resource_interval = resource_intervals[resource.id]
            for interval_index, interval in enumerate(resource_interval):
                if interval["duration"] < task.duration:
                    continue
                start, end = self._get_task_earliest_start_end(interval["slots"], task.duration, max_start_time)
                if start is not None:
                    resource_start_ends.append({
                        "resource": resource,
                        "interval_index": interval_index,
                        "start_time": start,
                        "end_time": end
                    })
        if not resource_start_ends:
            return None
        return min(resource_start_ends, key=lambda x: x['end_time']) 

    def _update_resource_intervals(self, resource_intervals, resource_id, interval_index, task_start, task_end):
        interval = resource_intervals(resource_id).pop(interval_index)
        new_intervals_slots = self._trim_and_split_interval(interval["slots"], task_start, task_end)
        for interval_slots in new_intervals_slots:
            if interval_slots:
                new_interval = self._create_resource_interval(interval_slots)
                resource_intervals(resource_id).insert(interval_index, new_interval)
        return resource_intervals
    
    def _get_task_earliest_start_end(self, intervals, task_duration, latest_start_time=0):
        remaining_duration = task_duration
        task_start = None
        for start, end in intervals:
            if latest_start_time >= end:
                continue

            if start <= latest_start_time <= end:
                task_start = latest_start_time
                start = latest_start_time

            slot_duration = end-start
            if slot_duration >= remaining_duration:
                task_end = end - (slot_duration - remaining_duration)
                return (task_start,task_end) 
            
            remaining_duration -= slot_duration
        return None, None

    def _trim_and_split_interval(self, slots, task_start, task_end):
        interval1 = []
        interval2 = []
        for start, end in slots:
            # append full intervals
            if task_start >= end:
                interval1.append((start,end))
                continue
            if task_end <= start:
                interval2.append((start,end))
                continue
            # trim intervals if between task_start and task_end
            if task_start > start:
                interval1.append((start, task_start))
            if task_end < end:
                interval2.append((task_end, end))
        return interval1, interval2

    def _get_resource_intervals(self, resources):
        return {resource.id : [self._create_resource_interval(resource.availability_slots)] for resource in resources}

    def _create_resource_interval(self, slots):
        return {
            "slots":slots,
            "duration": self._get_interval_duration(slots),
            "span": self._get_interval_start_end(slots)
        }
    
    def _get_interval_duration(self, slots):
        return sum(end - start for start, end in slots)
    
    def _get_interval_start_end(self, slots):
        return (slots[0][0], slots[-1][1])
    


