from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..models import Task


class TaskBatchProcessor:
    """
    The TaskBatchProcessor class is responsible for preprocessing tasks.
    """

    def __init__(self, task_graph: nx.DiGraph, task_dict: Dict[int, Task]):
        self.task_graph = task_graph
        self.task_dict = task_dict

    def split_tasks_into_batches(self) -> Dict[Tuple[int, int], Task]:
        # Perform a topological sort to get an ordering of nodes
        task_dict = deepcopy(self.task_dict)

        # Perform a topological sort to get an ordering of nodes
        task_order = list(nx.topological_sort(self.task_graph))

        # Iterate over the sorted tasks
        for task_id in task_order:
            task = self.task_dict[task_id]
            if task.batch_size > 0:
                # get successor tasks
                successor_tasks = [
                    self.task_dict[successor_id]
                    for successor_id in list(self.task_graph.successors(task_id))
                ]

                # split task
                task_splits = self._split_task_by_batch(
                    task=task, successor_tasks=successor_tasks
                )

                # update task_dict
                for split_task in task_splits:
                    task_dict[(split_task.id, split_task.batch_id)] = split_task

        return task_dict

    def _split_task_by_batch(
        self, task: Task, successor_tasks: Optional[List[Task]] = None
    ) -> List[Task]:
        """Splits a task into batches and updates the successor tasks' predecessors."""
        if task.batch_size is None or task.batch_size > task.quantity:
            return [task]

        batches = self._create_batches(task)

        # Update successors' predecessors
        if successor_tasks:
            unique_tasks = len(set(task.id for task in successor_tasks))
            # If all successors are the same task, assign each batch to a successor
            if len(batches) == len(successor_tasks) and unique_tasks == 1:
                for batch, task in zip(batches, successor_tasks):
                    task.predecessors = [batch]
            else:  # Else, assign all batches to all successors
                for task in successor_tasks:
                    task.predecessors = batches

        return batches

    def _create_batches(self, task: Task) -> List[Task]:
        """Creates batches of a task."""
        num_batches = task.quantity // task.batch_size
        remaining = task.quantity % task.batch_size
        batches = []

        for i in range(num_batches):
            new_task = self._create_new_task(task, i + 1, task.batch_size)
            # if i > 0:  # not the first batch, so predecessor is the previous batch
            #     new_task.predecessors = [batches[-1]]
            batches.append(new_task)

        if remaining > 0:
            new_task = self._create_new_task(task, num_batches + 1, remaining)
            # new_task.predecessors = [batches[-1]]  # predecessor is the previous batch
            batches.append(new_task)

        return batches

    def _create_new_task(self, task: Task, batch_id: int, quantity: int) -> Task:
        """Creates a new task with the given batch_id and quantity."""
        new_task = task.copy(deep=True)
        new_task.quantity = quantity
        new_task.batch_id = batch_id
        return new_task
