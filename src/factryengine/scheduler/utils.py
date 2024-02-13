from ..models import Task


def get_task_predecessors(task: Task, task_dict: dict) -> list[Task]:
    """
    returns a list of tasks that are predecessors of the given task
    """
    try:
        predecessors = [task_dict[pred_id] for pred_id in task.predecessor_ids]

    except KeyError as e:
        raise ValueError(f"Predecessor with ID {e.args[0]} does not exist.")
    return predecessors
