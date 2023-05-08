import networkx as nx
from intervaltree import IntervalTree

class HeuristicSolver():
    def __init__(self, tasks, resources):
        self.tasks = tasks
        self.resources = resources

    def solve(self):
        task_ids_ordered = self._get_task_order(self.tasks)
        task_dict = {task.id: task for task in self.tasks}
        task_vars = {task.id: {} for task in self.tasks}
        resource_interval_trees = self._get_resource_interval_trees(self.resources)

        unscheduled_tasks = []
        for i, task_id in enumerate(task_ids_ordered):
            task = task_dict[task_id]
            # get earliest start from max predecessors end
            task_earliest_start = self._get_task_earliest_start(task_vars, task)

            # find the resource who completes the task first
            fastest_resource = self._get_fastest_resource(task,resource_interval_trees, task_earliest_start)
            if fastest_resource is None:
                unscheduled_tasks.append(task.id)
                self._update_task_vars_unscheduled(task_vars, task)
                continue

            self._update_task_vars_scheduled(task_vars, task, fastest_resource)

            # update resource_intervals
            self._update_resource_interval_trees(
                resource_interval_trees= resource_interval_trees,
                resource_id= fastest_resource["resource"].id,
                interval_tree_index= fastest_resource["interval_index"],
                task_start= fastest_resource["task_start"],
                task_end= fastest_resource["task_end"]
            )

        return list(task_vars.values()) # Return values of the dictionary as a list
    
    def _update_task_vars_unscheduled(self, task_vars, task):
        task_values = {
            "task_id": task.id,
            "assigned_resource_id": None,
            "task_start": None,
            "task_end": None,
            "task_intervals": None
        }
        task_vars[task.id] = task_values

    def _update_task_vars_scheduled(self, task_vars, task, fastest_resource):
        task_values = {
            "task_id": task.id,
            "assigned_resource_id": fastest_resource["resource"].id,
            "task_start": fastest_resource["task_start"],
            "task_end": fastest_resource["task_end"],
            "task_intervals": [(interval.begin, interval.end) for interval in sorted(fastest_resource["task_interval_tree"])]
        }
        task_vars[task.id] = task_values

    def _get_task_earliest_start(self, task_vars, task):
        predecessor_max_end = max((task_vars[pred.id].get('task_end',0) for pred in task.predecessors), default=0)
        return predecessor_max_end + task.predecessor_delay

    def _get_task_order(self, tasks):
        """
        Returns a list of task IDs in the order required to complete them as quickly as possible while considering task priorities.

        :param tasks: List of tasks with durations, predecessors, and priorities.
        :return: List of task IDs in the order required to complete them as quickly as possible while considering task priorities.
        """
        task_graph = self._create_task_graph(tasks)
        longest_path = self._compute_longest_paths(task_graph)
        ordered_task_ids = self._custom_topological_sort(task_graph, tasks, longest_path)
        return ordered_task_ids

    def _create_task_graph(self, tasks):
        """
        Creates a directed acyclic graph from the given tasks.

        :param tasks: List of tasks with durations, predecessors, and priorities.
        :return: Directed acyclic graph representing task dependencies.
        """
        task_graph = nx.DiGraph()
        for task in tasks:
            task_graph.add_node(task.id, duration=task.duration, priority=task.priority)
            for predecessor in task.predecessors:
                task_graph.add_edge(predecessor.id, task.id)
        return task_graph

    def _compute_longest_paths(self, task_graph):
        """
        Computes the longest path for each node in the task graph using a topological sort.

        :param task_graph: Directed acyclic graph representing task dependencies.
        :return: Dictionary with task IDs as keys and the longest path for each task as values.
        """
        longest_path = {task.id: 0 for task in self.tasks}
        for task in nx.topological_sort(task_graph):
            duration = task_graph.nodes[task]['duration']
            for predecessor in task_graph.predecessors(task):
                longest_path[task] = max(longest_path[task], longest_path[predecessor] + duration)
        return longest_path

    def _custom_topological_sort(self, task_graph, tasks, longest_path):
        """
        Performs a custom topological sort of tasks considering priority and longest path.

        :param task_graph: Directed acyclic graph representing task dependencies.
        :param tasks: List of tasks with durations, predecessors, and priorities.
        :param longest_path: Dictionary with task IDs as keys and the longest path for each task as values.
        :return: List of task IDs in the order required to complete them as quickly as possible while considering task priorities.
        """
        visited = set()
        result = []

        def visit(node):
            """
            Recursive function to traverse the task graph in the desired order.
            """
            if node not in visited:
                visited.add(node)
                predecessors = sorted(
                    task_graph.predecessors(node),
                    key=lambda n: (task_graph.nodes[n]['priority'], -longest_path[n])
                )
                for predecessor in predecessors:
                    visit(predecessor)
                result.append(node)

        for task in sorted(tasks, key=lambda t: (task_graph.nodes[t.id]['priority'], -longest_path[t.id])):
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
                task_start, task_end, task_interval_tree = self._get_task_earliest_start_end(interval["slots"], task.duration, max_start_time)
                if task_start is not None:
                    resource_start_ends.append({
                        "resource": resource,
                        "interval_index": interval_index,
                        "task_start": task_start,
                        "task_end": task_end,
                        "task_interval_tree" : task_interval_tree 
                    })
        if not resource_start_ends:
            return None
        return min(resource_start_ends, key=lambda x: x['task_end']) 

    def _update_resource_interval_trees(self, resource_interval_trees, resource_id, interval_tree_index, task_start, task_end):
        interval_tree = resource_interval_trees[resource_id].pop(interval_tree_index)
        new_intervals_slots = self._chop_and_split_interval_tree(interval_tree["slots"], task_start, task_end)
        for interval_slots in new_intervals_slots:
            if interval_slots:
                new_interval = self._create_resource_interval_tree(interval_slots)
                resource_interval_trees[resource_id].insert(interval_tree_index, new_interval)
    
    def _get_task_earliest_start_end(self, interval_tree, task_duration, latest_start_time=0):
        remaining_duration = task_duration
        interval_tree.chop(0,latest_start_time)
        task_start = interval_tree.begin()
        for interval in sorted(interval_tree):
            start, end = interval.begin, interval.end
            interval_duration = end-start
            if interval_duration >= remaining_duration:
                task_end = end - (interval_duration - remaining_duration)
                task_interval_tree = self._get_task_interval_tree(interval_tree, task_start, task_end)
                return (task_start, task_end, task_interval_tree) 
            remaining_duration -= interval_duration
        return (None, None, None)
    
    def _get_task_interval_tree(self, interval_tree, task_start, task_end):
        interval_tree.slice(task_start)
        interval_tree.slice(task_end)
        return interval_tree[task_start: task_end]
    
    def _chop_and_split_interval_tree(self, interval_tree, first_point, second_point):
        interval_tree.slice(first_point)
        interval_tree.slice(second_point)
        return interval_tree[:first_point], interval_tree[second_point:]

    def _get_resource_interval_trees(self, resources):
        return {resource.id : [self._create_resource_interval_tree(resource.availability_slots)] for resource in resources}

    def _create_resource_interval_tree(self, availability_slots):
        interval_tree = self._availability_slots_to_interval_tree(availability_slots)
        return {
            "slots": interval_tree,
            "duration": self._get_interval_tree_duration(interval_tree),
            "span": self._get_interval_tree_start_end(interval_tree)
        }
    
    def _availability_slots_to_interval_tree(self, availability_slots):
        return IntervalTree.from_tuples(availability_slots)

    def _get_interval_tree_duration(self, interval_tree):
        return sum(interval.end - interval.begin for interval in interval_tree)
    
    def _get_interval_tree_start_end(self, interval_tree):
        return (interval_tree.begin(), interval_tree.end())
    


