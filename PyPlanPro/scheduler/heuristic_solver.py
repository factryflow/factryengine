import networkx as nx
import copy
from intervaltree import IntervalTree

class TaskManager:
    """
    A class used to manage tasks in a task list.
    """
    def __init__(self, tasks):
        """
        Initializes the TaskManager with a list of tasks.
        """
        self.tasks = tasks
        self.task_dict = {task.id: task for task in tasks}

    def get_task_earliest_start(self, task_vars, task):
        """
        Determines the earliest possible start time for a task based on its predecessors.
        """
        if not task.predecessors:
            return 0

        task_ends = [task_vars[pred.id].get('task_end') for pred in task.predecessors]
        if None in task_ends:
            return None

        return max(task_ends) + task.predecessor_delay

    def get_fastest_resource(self, task, resource_interval_trees, max_start_time):
        """
        Returns the resource with the earliest end time for a task.
        """
        resource_start_ends = self._calculate_resource_start_ends(task, resource_interval_trees, max_start_time)
        if not resource_start_ends:
            return None
        return min(resource_start_ends, key=lambda x: x['task_end']) 

    def _calculate_resource_start_ends(self, task, resource_interval_trees, max_start_time):
        """
        Calculates and returns start and end times for all resources for a task.
        """
        resource_start_ends = []
        for resource in task.get_resources():
            task_duration = self._calculate_task_duration(task, resource)
            interval_trees = resource_interval_trees[resource.id]
            
            for index, interval in enumerate(interval_trees):
                if interval["duration"] < task_duration:
                    continue
                
                start_end = self._get_task_earliest_start_end(interval["slots"], task_duration, max_start_time)
                if start_end[0] is not None:
                    resource_start_ends.append({
                        "resource": resource,
                        "interval_index": index,
                        "task_start": start_end[0],
                        "task_end": start_end[1],
                        "task_interval_tree" : start_end[2] 
                    })
        return resource_start_ends
    
    def _get_task_earliest_start_end(self, slots, task_duration, latest_start_time=0):
        """
        Returns the earliest start and end time for a task given its duration and the latest start time.
        """
        interval_tree = self._prepare_interval_tree(slots, latest_start_time)
        remaining_duration = task_duration
        task_start = interval_tree.begin()
        
        for interval in sorted(interval_tree):
            start, end = interval.begin, interval.end
            interval_duration = end-start
            
            if interval_duration >= remaining_duration:
                task_end = end - (interval_duration - remaining_duration)
                scheduled_intervals = self._get_task_scheduled_intervals(interval_tree, task_start, task_end)
                return (task_start, task_end, scheduled_intervals) 
            remaining_duration -= interval_duration
        return (None, None, None)
    
    def _get_task_scheduled_intervals(self, interval_tree, task_start, task_end):
        """
        Slices the interval tree and returns the intervals that cover the task's execution time.
        """
        interval_tree.slice(task_start)
        interval_tree.slice(task_end)
        return interval_tree[task_start: task_end]
    
    def _calculate_task_duration(self, task ,resource):
        """
        Calculates and returns the duration of a task with respect to a resource.
        """
        return task.duration / (task.resource_group.efficiency_multiplier * resource.efficiency_multiplier)
    
    @staticmethod
    def _prepare_interval_tree(slots, latest_start_time):
        """
        Prepares and returns a copy of the interval tree, trimmed from the start to the latest start time.
        """
        interval_tree = copy.deepcopy(slots)
        interval_tree.chop(0, latest_start_time)
        return interval_tree

class TaskGraph:
    def __init__(self, tasks):
        self.tasks = {task.id: task for task in tasks}
        self.graph = self._create_task_graph()

    def _create_task_graph(self):
        """
        Creates a directed acyclic graph from the given tasks.
        """
        task_graph = nx.DiGraph()
        for task in self.tasks.values():
            task_graph.add_node(task.id, duration=task.duration, priority=task.priority)
            for predecessor in task.predecessors:
                task_graph.add_edge(predecessor.id, task.id)
        return task_graph

    def _compute_longest_paths(self):
        """
        Computes the longest path for each node in the task graph using a topological sort.
        """
        longest_path = {task_id: 0 for task_id in self.tasks}
        for task in nx.topological_sort(self.graph):
            duration = self.graph.nodes[task]['duration']
            for predecessor in self.graph.predecessors(task):
                longest_path[task] = max(longest_path[task], longest_path[predecessor] + duration)
        return longest_path

    def _custom_topological_sort(self, longest_path):
        """
        Performs a custom topological sort of tasks considering priority and longest path.
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
                    self.graph.predecessors(node),
                    key=lambda n: (self.graph.nodes[n]['priority'], -longest_path[n])
                )
                for predecessor in predecessors:
                    visit(predecessor)
                result.append(node)

        for task in sorted(self.tasks.values(), key=lambda t: (self.graph.nodes[t.id]['priority'], -longest_path[t.id])):
            visit(task.id)

        return result

    def get_task_order(self):
        """
        Returns a list of task IDs in the order required to complete them as quickly as possible while considering task priorities.
        """
        longest_path = self._compute_longest_paths()
        return self._custom_topological_sort(longest_path)

class IntervalTreeManager:
    def __init__(self, resources):
        self.resources = resources

    def get_resource_interval_trees(self):
        return {resource.id : [self._create_resource_interval_tree(resource.availability_slots)] for resource in self.resources}

    def _create_resource_interval_tree(self, availability_slots):
        interval_tree = self._availability_slots_to_interval_tree(availability_slots)
        return {
            "slots": interval_tree,
            "duration": self._get_interval_tree_duration(interval_tree),
            "span": self._get_interval_tree_start_end(interval_tree)
        }

    def update_resource_interval_trees(self, resource_interval_trees, resource_id, interval_tree_index, task_start, task_end):
        interval_tree = resource_interval_trees[resource_id].pop(interval_tree_index)
        new_intervals_slots = self._chop_and_split_interval_tree(interval_tree["slots"], task_start, task_end)
        for interval_slots in new_intervals_slots:
            if interval_slots:
                new_interval = self._create_resource_interval_tree(interval_slots)
                resource_interval_trees[resource_id].insert(interval_tree_index, new_interval)
    
    def _chop_and_split_interval_tree(self, interval_tree, first_point, second_point):
        interval_tree.slice(first_point)
        interval_tree.slice(second_point)
        return interval_tree[:first_point], interval_tree[second_point:]
    
    def _availability_slots_to_interval_tree(self, availability_slots):
        return IntervalTree.from_tuples(availability_slots)

    def _get_interval_tree_duration(self, interval_tree):
        return sum(interval.end - interval.begin for interval in interval_tree)
    
    def _get_interval_tree_start_end(self, interval_tree):
        return (interval_tree.begin(), interval_tree.end())
    
class HeuristicSolver():
    def __init__(self, tasks, resources):
        self.task_manager = TaskManager(tasks)
        self.task_graph = TaskGraph(tasks)
        self.interval_tree_manager = IntervalTreeManager(resources)
        self.resource_interval_trees = self.interval_tree_manager.get_resource_interval_trees()
        self.task_vars = {task.id : {"assigned_resource_id": None, "task_start": None, "task_end": None, "task_intervals": None} for task in tasks}

    def solve(self):
        task_order = self.task_graph.get_task_order()
        unscheduled_tasks = []

        for task_id in task_order:
            task = self.task_manager.task_dict[task_id]
            # get earliest start from max predecessors end
            task_earliest_start = self.task_manager.get_task_earliest_start(self.task_vars, task)
            if task_earliest_start is None:
                unscheduled_tasks.append(task_id)
                continue

            # find the resource who completes the task first
            resource_data = self.task_manager.get_fastest_resource(task, self.resource_interval_trees, task_earliest_start)
            if resource_data is None:
                unscheduled_tasks.append(task_id)
                continue
            
            self._update_task_vars_scheduled(task_id, resource_data)

            # update resource_intervals
            self.interval_tree_manager.update_resource_interval_trees(
                resource_interval_trees= self.resource_interval_trees,
                resource_id= resource_data["resource"].id,
                interval_tree_index= resource_data["interval_index"],
                task_start= resource_data["task_start"],
                task_end= resource_data["task_end"]
            )

        return list(self.task_vars.values()) # Return values of the dictionary as a list

    def _update_task_vars_scheduled(self, task_id, resource_data):
        task_values = {
            "task_id": task_id,
            "assigned_resource_id": resource_data["resource"].id,
            "task_start": resource_data["task_start"],
            "task_end": resource_data["task_end"],
            "task_intervals": [(interval.begin, interval.end) for interval in sorted(resource_data["task_interval_tree"])]
        }
        self.task_vars[task_id] = task_values

    
         

    


