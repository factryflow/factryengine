from .heuristic_solver.main import HeuristicSolver
from .scheduler_result import SchedulerResult

class Scheduler():
    def __init__(self, tasks):
        self.tasks = tasks
        self.resources = set([resource for task in tasks for constraint in task.constraints for resource in constraint.get_resources()])
    
    def schedule(self):
        heuristic_solver = HeuristicSolver(self.tasks, self.resources)
        result = heuristic_solver.solve()

        scheduler_result = SchedulerResult(result)
        print(scheduler_result.summary())
        return scheduler_result
