from ..models import Resource, Task

class TaskSplitter:
    """
    The TaskSplitter class is responsible for splitting tasks into batches.
    """

    def __init__(self, task: Task, batch_size: int):
        self.task = task
        self.batch_size = batch_size

    def split_into_batches(self) -> list[Task]:
        """
        Splits a task into batches.
        """
        num_batches, remaining = divmod(self.task.quantity, self.batch_size)
        batches = [
            self._create_new_task(i + 1, self.batch_size)
            for i in range(num_batches)
        ]

        if remaining > 0:
            batches.append(self._create_new_task(num_batches + 1, remaining))

        return batches

    def _create_new_task(self, batch_id: int, quantity: int) -> Task:
        """Creates a new task with the given batch_id and quantity."""
        new_task = self.task.model_copy(deep=True)
        new_task.quantity = quantity
        new_task.duration = (quantity / self.task.quantity) * self.task.duration
        new_task.set_batch_id(batch_id)
        return new_task
