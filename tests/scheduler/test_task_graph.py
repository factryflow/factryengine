import pytest
from planbee import Task
from planbee.scheduler.task_graph import TaskGraph

# Assuming a Resource class that can be initialized as Resource(id)


@pytest.fixture
def tasks_dict() -> dict[str, Task]:
    task1 = Task(id=1, duration=2, priority=1, resources=[], predecessors=[])
    task2 = Task(id=2, duration=1, priority=2, resources=[], predecessors=[task1])
    task3 = Task(
        id=3, duration=3, priority=3, resources=[], predecessors=[task1, task2]
    )
    tasks_dict = {
        "1": task1,
        "2": task2,
        "3": task3,
    }

    return tasks_dict


def test_create_task_graph(tasks_dict):
    task_graph = TaskGraph(tasks_dict)

    assert len(task_graph.graph) == 3
    assert len(task_graph.graph.edges) == 3
    assert "1" in task_graph.graph
    assert "2" in task_graph.graph
    assert "3" in task_graph.graph


def test_compute_longest_paths(tasks_dict):
    task_graph = TaskGraph(tasks_dict)
    longest_paths = task_graph._compute_longest_paths()

    assert longest_paths == {"1": 0, "2": 1, "3": 4}


def test_custom_topological_sort(tasks_dict):
    task_graph = TaskGraph(tasks_dict)
    longest_paths = task_graph._compute_longest_paths()
    sorted_tasks = task_graph._custom_topological_sort(longest_paths)

    assert sorted_tasks == ["1", "2", "3"]


def test_get_task_order(tasks_dict):
    task_graph = TaskGraph(tasks_dict)
    task_order = task_graph.get_task_order()

    assert task_order == ["1", "2", "3"]