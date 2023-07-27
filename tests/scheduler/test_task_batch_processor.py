import networkx as nx
import pytest
from planbee import Resource, Task
from planbee.scheduler.task_batch_processor import TaskBatchProcessor
from planbee.scheduler.task_graph import TaskGraph


# Replace the following line with actual task and graph creation
def create_task(id, quantity, batch_size, duration=10):
    resource = Resource(id=1, available_windows=[(1, 100)])
    return Task(
        id=id,
        duration=duration,
        priority=1,
        quantity=quantity,
        batch_size=batch_size,
        resources=resource,
    )


def create_graph():
    G = nx.DiGraph()
    # Add edges and nodes as needed
    return G


@pytest.fixture
def task_processor():
    task_dict = {1: create_task(1, 50, 10), 2: create_task(2, 100, 20)}
    task_graph = create_graph()
    return TaskBatchProcessor(task_graph, task_dict)


test_case_values = [
    (
        "no batch size",
        {"1": create_task(1, 50, None)},
        {"1": create_task(1, 50, None)},
    ),
    (
        "no quantity",
        {"1": create_task(1, None, 10)},
        {"1": create_task(1, None, 10)},
    ),
    (
        "no batch size or quantity",
        {"1": create_task(1, None, None)},
        {"1": create_task(1, None, None)},
    ),
    (
        "batch size > quantity",
        {"1": create_task(1, 10, 50)},
        {"1": create_task(1, 10, 50)},
    ),
    (
        "batch size = quantity",
        {"1": create_task(1, 50, 50)},
        {"1": create_task(1, 50, 50)},
    ),
    (
        "create two batches",
        {"1": create_task(1, 20, 10, 10)},
        {"1-1": create_task(1, 10, 10, 5), "1-2": create_task(1, 10, 10, 5)},
    ),
    (
        "create two batches with remainder",
        {"1": create_task(1, 19, 10, 19)},
        {"1-1": create_task(1, 10, 10, 10), "1-2": create_task(1, 9, 10, 9)},
    ),
]


@pytest.mark.parametrize(
    "test_name, task_dict, expected",
    test_case_values,
    ids=[case[0] for case in test_case_values],
)
def test_split_tasks_into_batches(test_name, task_dict, expected):
    graph = TaskGraph(task_dict).graph
    task_processor = TaskBatchProcessor(task_graph=graph, task_dict=task_dict)
    result = task_processor.split_tasks_into_batches()

    assert len(result) == len(expected), "The number of tasks is not equal"

    for key in expected.keys():
        assert key in result, f"Key {key} not in result"
        assert result[key].id == expected[key].id
        assert result[key].duration == expected[key].duration
        assert result[key].priority == expected[key].priority
        assert result[key].quantity == expected[key].quantity
        assert result[key].batch_size == expected[key].batch_size


def get_task_predecessor_uids(task):
    return [pred.uid for pred in task.predecessors]


def test_split_tasks_into_batches_predecessors():
    task_dict = {
        "1": create_task(1, 20, 10),
        "2": create_task(2, 100, 50),
        "3": create_task(3, 100, 50),
    }
    task_dict["2"].predecessors.append(task_dict["1"])
    task_dict["3"].predecessors.append(task_dict["2"])

    graph = TaskGraph(task_dict).graph
    task_processor = TaskBatchProcessor(task_graph=graph, task_dict=task_dict)
    result = task_processor.split_tasks_into_batches()

    assert (
        len(result) == 6
    ), "The length of the result does not match the expected value."
    assert (
        len(result["1-1"].predecessors) == 0
    ), "The length of predecessors of task '1-1' is not as expected."
    assert get_task_predecessor_uids(result["2-1"]) == [
        "1-1",
        "1-2",
    ], "The predecessors of task '2-1' do not match the expected value."
    assert get_task_predecessor_uids(result["2-2"]) == [
        "1-1",
        "1-2",
    ], "The predecessors of task '2-2' do not match the expected value."
    assert get_task_predecessor_uids(result["3-1"]) == [
        "2-1"
    ], "The predecessors of task '3-1' do not match the expected value."
    assert get_task_predecessor_uids(result["3-2"]) == [
        "2-2"
    ], "The predecessors of task '3-2' do not match the expected value."
