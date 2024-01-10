import pytest
from factryengine.models import Task
from factryengine.scheduler.utils import get_task_predecessors


@pytest.fixture
def tasks_dict() -> dict[str, Task]:
    task1 = Task(id=1, duration=2, priority=1, resources=[], predecessors=[])
    task2 = Task(id=2, duration=1, priority=2, resources=[], predecessor_ids=[1])
    task3 = Task(id=3, duration=3, priority=3, resources=[], predecessor_ids=[1, 2])
    tasks_dict = {
        "1": task1,
        "2": task2,
        "3": task3,
    }

    return tasks_dict


@pytest.mark.parametrize(
    "task_id,expected_predecessors",
    [
        ("1", []),
        ("2", ["1"]),
        ("3", ["1", "2"]),
    ],
)
def test_can_get_task_predecessors(task_id, expected_predecessors, tasks_dict):
    task = tasks_dict[task_id]
    predecessors = get_task_predecessors(task, tasks_dict)

    assert [pred.get_id() for pred in predecessors] == expected_predecessors


def test_get_task_predecessors_raises_error_if_predecessor_does_not_exist(tasks_dict):
    task = Task(id=4, duration=2, priority=1, resources=[], predecessor_ids=[5])
    with pytest.raises(ValueError):
        predecessors = get_task_predecessors(task, tasks_dict)
