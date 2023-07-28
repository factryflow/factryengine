from copy import deepcopy

import networkx as nx

from ..models import Task


class TaskBatchProcessor:
    """
    The TaskBatchProcessor class is responsible for preprocessing tasks.
    """

    def __init__(self, task_graph: nx.DiGraph, task_dict: dict[str, Task]):
        self.task_graph = task_graph
        self.task_dict = deepcopy(task_dict)

    def split_tasks_into_batches(self) -> dict[str, Task]:
        """
        This function performs splitting of tasks into batches if necessary and returns
        an updated task dictionary with possibly split tasks. Tasks are split only if
        they have a batch size and the quantity is greater than the batch size.
        """
        # get the order of tasks based on their dependencies
        task_order = list(nx.topological_sort(self.task_graph))

        for task_uid in task_order:
            current_task = self.task_dict[task_uid]

            # Skip the current iteration if the task doesn't have a batch size or qty
            if not current_task.is_splittable():
                continue

            # split current task into batches
            task_splits = TaskSplitter(task=current_task).split_into_batches()

            # update the relationships of predecessor and successor tasks
            self._update_predecessor_successor_relationships(current_task, task_splits)

            # remove the current task from the task dictionary
            del self.task_dict[task_uid]

            # add the split tasks to the task dictionary
            for split_task in task_splits:
                self.task_dict[split_task.uid] = split_task

        return self.task_dict

    def _update_predecessor_successor_relationships(
        self, task: Task, batches: list[Task]
    ) -> None:
        """
        This function updates the predecessor and successor relationships of the given
        task
        """

        # update predecessors
        if self._are_predecessors_single_task(task):
            for batch, pred in zip(batches, task.predecessors):
                batch.predecessors = [pred]

        # update successors
        successor_tasks = self._get_successor_tasks(task.uid)
        for successor in successor_tasks:
            successor.predecessors.remove(task)
            successor.predecessors.extend(batches)

    def _are_predecessors_single_task(self, task: Task) -> bool:
        """Checks if the predecessors of the given task are a single task"""
        unique_tasks = len(set(pred_task.id for pred_task in task.predecessors))
        predecessors_quantity = sum(
            pred_task.quantity for pred_task in task.predecessors
        )
        return unique_tasks == 1 and predecessors_quantity == task.quantity

    def _get_successor_tasks(self, task_uid: str) -> list[Task]:
        """
        Given a task_uid, this function returns the list of successor tasks.
        """
        successor_uids = list(self.task_graph.successors(task_uid))
        return [self.task_dict[successor_id] for successor_id in successor_uids]


class TaskSplitter:
    """
    The TaskSplitter class is responsible for splitting tasks into batches.
    """

    def __init__(self, task: Task):
        self.task = task

    def split_into_batches(self) -> list[Task]:
        """
        Splits a task into batches.
        """
        num_batches, remaining = divmod(self.task.quantity, self.task.batch_size)
        batches = [
            self._create_new_task(i + 1, self.task.batch_size)
            for i in range(num_batches)
        ]

        if remaining > 0:
            batches.append(self._create_new_task(num_batches + 1, remaining))

        return batches

    def _create_new_task(self, batch_id: int, quantity: int) -> Task:
        """Creates a new task with the given batch_id and quantity."""
        new_task = self.task.copy(deep=True)
        new_task.quantity = quantity
        new_task.duration = (quantity / self.task.quantity) * self.task.duration
        new_task.set_batch_id(batch_id)
        return new_task
