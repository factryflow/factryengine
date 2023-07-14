import pytest
from planbee import Task
from planbee.scheduler.task_graph import TaskGraph

# Assuming a Resource class that can be initialized as Resource(id)


@pytest.fixture
def tasks():
    task1 = Task(id=1, duration=2, priority=1, resources=[], predecessors=[])
    task2 = Task(id=2, duration=1, priority=2, resources=[], predecessors=[task1])
    task3 = Task(
        id=3, duration=3, priority=3, resources=[], predecessors=[task1, task2]
    )
    return [task1, task2, task3]


def test_create_task_graph(tasks):
    task_graph = TaskGraph(tasks)

    assert len(task_graph.graph) == 3
    assert len(task_graph.graph.edges) == 3
    assert 1 in task_graph.graph
    assert 2 in task_graph.graph
    assert 3 in task_graph.graph


def test_compute_longest_paths(tasks):
    task_graph = TaskGraph(tasks)
    longest_paths = task_graph._compute_longest_paths()

    assert longest_paths == {1: 0, 2: 1, 3: 4}


def test_custom_topological_sort(tasks):
    task_graph = TaskGraph(tasks)
    longest_paths = task_graph._compute_longest_paths()
    sorted_tasks = task_graph._custom_topological_sort(longest_paths)

    assert sorted_tasks == [1, 2, 3]


def test_get_task_order(tasks):
    task_graph = TaskGraph(tasks)
    task_order = task_graph.get_task_order()

    assert task_order == [1, 2, 3]
