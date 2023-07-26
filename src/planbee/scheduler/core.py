from ..models import Task
from .heuristic_solver.main import HeuristicSolver
from .scheduler_result import SchedulerResult
from .task_batch_processor import TaskBatchProcessor
from .task_graph import TaskGraph


class Scheduler:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        self.resources = set(
            resource for task in tasks for resource in task.get_resources()
        )
        self.task_dict = self.get_task_dict(tasks)
        self.task_graph = TaskGraph(self.task_dict)

    def schedule(self):
        # merge intervals for all resources
        for resource in self.resources:
            resource.merge_intervals()

        # schedule tasks
        task_order = self.task_graph.get_task_order()
        heuristic_solver = HeuristicSolver(self.task_dict, self.resources, task_order)
        result = heuristic_solver.solve()

        scheduler_result = SchedulerResult(result)
        print(scheduler_result.summary())
        return scheduler_result

    def get_task_dict(self, tasks: list[Task]):
        """
        returns the task dictionary with tasks split into batches
        """
        task_dict = {task.uid: task for task in tasks}
        task_graph = TaskGraph(task_dict).graph
        task_batch_processor = TaskBatchProcessor(task_graph, task_dict)
        task_dict_with_batches = task_batch_processor.split_tasks_into_batches()
        return task_dict_with_batches
