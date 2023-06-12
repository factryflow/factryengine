import networkx as nx
from ...models.availability_slot import UnscheduledQueue

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
    
    def get_fastest_team(self, task, resource_availability_slots):
        resource_combinations = task.get_unique_resource_combinations()

        # for combination in resource_combinations:
            # find earliest end_date
            # given list of availability slots, find common end date 

    def get_fastest_resources(self, task, resource_availability_slots, max_start_time):
        """
        Returns the resource with the earliest end time for a task.
        """
        resource_start_ends = self._calculate_resource_start_ends(task, resource_availability_slots, max_start_time)
        if not resource_start_ends:
            return None
        return min(resource_start_ends, key=lambda x: x['task_end']) 

    # def get_resource_combinations(self, task):

    def _calculate_resource_start_ends(self, task, resource_availability_slots, max_start_time):
        """
        Calculates and returns start and end times for all resources for a task.
        """
        resource_start_ends = []
        for resource in task.get_resources():
            task_duration = self._calculate_task_duration(task, resource)
            availability_slot = resource_availability_slots[resource.id]
            
            for index, availability_slot in enumerate(availability_slot):
                if availability_slot.duration < task_duration:
                    continue
                
                start_end = self._get_resource_earliest_start_end(availability_slot, task_duration, max_start_time)
                if start_end[0] is not None:
                    resource_start_ends.append({
                        "resource": resource,
                        "interval_index": index,
                        "task_start": start_end[0],
                        "task_end": start_end[1],
                        "task_availability_slot" : start_end[2] 
                    })
        return resource_start_ends
    
    def _get_resource_earliest_start_end(self, availability_slot, task_duration, latest_start_time=None):
        """
        Returns the earliest start and end time for a task given its duration and the latest start time.
        """
        availability_slot = availability_slot[latest_start_time:] if latest_start_time > 0 else availability_slot
        remaining_duration = task_duration
        task_start = availability_slot.begin()
        
        for interval in sorted(availability_slot.tree):
            start, end = interval.begin, interval.end
            interval_duration = end-start
            
            if interval_duration >= remaining_duration:
                task_end = end - (interval_duration - remaining_duration)
                scheduled_intervals = availability_slot[task_start: task_end]
                return (task_start, task_end, scheduled_intervals) 
            remaining_duration -= interval_duration
        return (None, None, None)
    
    def _calculate_task_duration(self, task , resource):
        """
        Calculates and returns the duration of a task with respect to a resource.
        """
        return task.duration #/ (task.resource_group.efficiency_multiplier * resource.efficiency_multiplier)

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

class AvailabilitySlotManager:
    def __init__(self, resources):
        self.resources = resources

    def create_resource_unscheduled_queue_dict(self):
        return {resource.id : UnscheduledQueue.from_tuples(resource.availability_slots) for resource in self.resources}

    def update_resource_availability_slots(self, resource_availability_slots, resource_id, availability_slot_index, task_start, task_end):
        availability_slot = resource_availability_slots[resource_id].pop(availability_slot_index)
        new_availability_slots = availability_slot[:task_start], availability_slot[task_end:]
        for availability_slot in new_availability_slots:
            if availability_slot:
                resource_availability_slots[resource_id].insert(availability_slot_index, availability_slot)
                
class HeuristicSolver():
    def __init__(self, tasks, resources):
        self.task_manager = TaskManager(tasks)
        self.task_graph = TaskGraph(tasks)
        self.resource_windows = self._create_resource_unscheduled_queue_dict()
        self.task_vars = {task.id : {"assigned_resource_id": None, "task_start": None, "task_end": None, "task_intervals": None} for task in tasks}
        self.task_dict = {task.id: task for task in tasks}

    def solve(self):
        task_order = self.task_graph.get_task_order()
        unscheduled_tasks = []

        for task_id in task_order:
            task = self.task_dict[task_id]
            # get earliest start from max predecessors end
            task_earliest_start = self.task_manager.get_task_earliest_start(self.task_vars, task)
            if task_earliest_start is None:
                unscheduled_tasks.append(task_id)
                continue

            # find the resource who completes the task first
            resource_data = self.task_manager.get_fastest_resources(task, self.resource_windows, task_earliest_start)
            if resource_data is None:
                unscheduled_tasks.append(task_id)
                continue
            
            self._update_task_vars_scheduled(task_id, resource_data)

            # update resource_intervals
            self.availability_slot_manager.update_resource_availability_slots(
                resource_availability_slots= self.resource_windows,
                resource_id= resource_data["resource"].id,
                availability_slot_index= resource_data["interval_index"],
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
            "task_intervals": [(interval.begin, interval.end) for interval in sorted(resource_data["task_availability_slot"].tree)]
        }
        self.task_vars[task_id] = task_values

    def _create_resource_unscheduled_queue_dict(self):
        return {resource.id : UnscheduledQueue.from_tuples(resource.availability_slots) for resource in self.resources}
    
         

    


