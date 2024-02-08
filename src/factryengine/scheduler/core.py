from ..models import Resource, Task
from .heuristic_solver.main import HeuristicSolver
from .scheduler_result import SchedulerResult
from .task_graph import TaskGraph


class Scheduler:
    def __init__(self, tasks: list[Task], resources: list[Resource]):
        self.tasks = tasks
        self.resources = resources
        self.task_dict = self.get_task_dict(tasks)
        self.task_graph = TaskGraph(self.task_dict)

    def schedule(self) -> SchedulerResult:
        """
        Schedule tasks based on the task order from the task graph.
        """

        # Merge intervals for all resources
        for resource in self.resources:
            resource.merge_intervals()

        # Get the order in which tasks should be scheduled
        task_order = self.task_graph.get_task_order()

        # Create a heuristic solver with the tasks, resources, and task order
        heuristic_solver = HeuristicSolver(
            task_dict=self.task_dict, resources=self.resources, task_order=task_order
        )

        # Use the heuristic solver to find a solution
        solver_result = heuristic_solver.solve()

        # Create a scheduler result with the solver result and unscheduled tasks
        scheduler_result = SchedulerResult(
            task_vars=solver_result,
            unscheduled_task_ids=heuristic_solver.unscheduled_task_ids,
        )

        # Print a summary of the scheduling results
        print(scheduler_result.summary())

        # Return the scheduler result
        return scheduler_result

    def get_task_dict(self, tasks: list[Task]):
        """
        returns the task dictionary with task id as key and task object as value
        """
        return {task.get_id(): task for task in tasks}
