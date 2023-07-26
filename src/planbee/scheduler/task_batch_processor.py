from copy import deepcopy
from typing import Dict, List, Optional

import networkx as nx

from ..models import Task


class TaskBatchProcessor:
    """
    The TaskBatchProcessor class is responsible for preprocessing tasks.
    """

    def __init__(self, task_graph: nx.DiGraph, task_dict: Dict[str, Task]):
        self.task_graph = task_graph
        self.task_dict = task_dict

    def split_tasks_into_batches(self) -> Dict[int | str, Task]:
        """
        This function performs splitting of tasks into batches if necessary and returns
        an updated task dictionary with possibly split tasks. Tasks are split only if
        they have a batch size and the quantity is greater than the batch size.
        """
        task_dict_copy = deepcopy(self.task_dict)

        # get the order of tasks based on their dependencies
        task_order = list(nx.topological_sort(self.task_graph))

        for task_uid in task_order:
            current_task = self.task_dict[task_uid]
            # Skip the current iteration if the task doesn't have a batch size or qty
            if current_task.batch_size is None or current_task.quantity is None:
                continue

            # Check if the task needs to be split into batches
            if (
                current_task.batch_size > 0
                and current_task.quantity > current_task.batch_size
            ):
                successor_tasks = self._get_successor_tasks(task_uid)

                # split current task into batches
                task_splits = self._split_task_by_batch(
                    task=current_task, successor_tasks=successor_tasks
                )

                # remove the current task from the task dictionary
                del task_dict_copy[task_uid]

                # add the split tasks to the task dictionary
                for split_task in task_splits:
                    task_dict_copy[split_task.uid] = split_task

        return task_dict_copy

    def _get_successor_tasks(self, task_uid):
        """
        Given a task_uid, this function returns the list of successor tasks.
        """
        successor_ids = list(self.task_graph.successors(task_uid))
        return [self.task_dict[successor_id] for successor_id in successor_ids]

    def _split_task_by_batch(
        self, task: Task, successor_tasks: Optional[List[Task]] = None
    ) -> List[Task]:
        """
        Splits a task into batches and updates the successor tasks' predecessors.
        """

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
        new_task.duration = (quantity / task.quantity) * task.duration
        new_task.set_batch_id(batch_id)
        return new_task
