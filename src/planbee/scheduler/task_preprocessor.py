from typing import Optional

import networkx as nx

from ..models import Task


class TaskPreprocessor:
    """
    The TaskPreprocessor class is responsible for preprocessing tasks.
    """

    # def __init__(tasks: list[Task]):
    #     self.tasks = tasks
    @staticmethod
    def split_all_tasks(task_graph: nx.DiGraph) -> dict[str, list[Task]]:
        # Perform a topological sort to get an ordering of nodes
        task_order = list(nx.topological_sort(task_graph))

        # Dict to hold all the split tasks
        split_tasks = {}

        # Iterate over the sorted tasks
        for task_id in task_order:
            task = task_graph.nodes[task_id]["task"]

            # Get the successor tasks
            successor_ids = list(task_graph.successors(task_id))
            successor_tasks = [
                split_tasks[successor_id][0]
                for successor_id in successor_ids
                if successor_id in split_tasks
            ]

            # Split the task and store it in the dictionary
            split_tasks[task_id] = TaskPreprocessor.split_task_by_batch(
                task, successor_tasks
            )

        return split_tasks

    @staticmethod
    def split_task_by_batch(
        task: Task, successor_tasks: Optional[list[Task]] = None
    ) -> list[Task]:
        # Check if batch_size is defined and less than quantity
        if task.batch_size and task.batch_size < task.quantity:
            num_batches = task.quantity // task.batch_size
            remaining = task.quantity % task.batch_size
            # Create batches
            batches = []
            for i in range(num_batches):
                new_task = task.copy(deep=True)
                new_task.quantity = task.batch_size
                new_task.id = f"{task.id}_batch_{i+1}"
                if i > 0:  # not the first batch, so predecessor is the previous batch
                    new_task.predecessors = [batches[-1]]
                batches.append(new_task)
            # Handle remaining quantity
            if remaining > 0:
                new_task = task.copy(deep=True)
                new_task.quantity = remaining
                new_task.id = f"{task.id}_batch_{num_batches + 1}"
                new_task.predecessors = [
                    batches[-1]
                ]  # predecessor is the previous batch
                batches.append(new_task)
            # Update successors' predecessors
            if successor_tasks:
                # If the number of batches is the same, assign corresponding batches
                if len(batches) == len(successor_tasks):
                    for batch, task in zip(batches, successor_tasks):
                        task.predecessors = [batch]
                else:  # Else, assign all batches to all successors
                    for task in successor_tasks:
                        task.predecessors = batches
            return batches
        else:
            return [task]
