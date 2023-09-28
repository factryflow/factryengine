import numpy as np
import pytest

from factryengine.scheduler.heuristic_solver.task_allocator import TaskAllocator


@pytest.fixture
def task_allocator():
    return TaskAllocator()


def test_allocate_task_returns_expected_result(task_allocator):
    resource_windows = [np.array([[0, 1, 1, -1], [2, 3, 1, 0]])]
    resource_ids = np.array([1])
    task_duration = 2
    resource_count = 1
    resource_group_indices = [[0]]
    result = task_allocator.allocate_task(
        resource_windows=resource_windows,
        resource_ids=resource_ids,
        task_duration=task_duration,
        resource_count=resource_count,
        resource_group_indices=resource_group_indices,
    )
    # Check if the result is a dictionary
    assert isinstance(result, dict)
    # Check if the result has the correct keys (allocated resources)
    assert list(result.keys()) == [1]
    # Validate a specific element in the result (replace with your expected values)
    assert result == {1: [(0.0, 1.0), (2.0, 3.0)]}


def test_allocate_task_returns_none_when_no_solution(task_allocator):
    resource_windows = [np.array([[0, 1, 1, -1], [2, 3, 1, 0]])]
    resource_ids = np.array([1])
    task_duration = 4
    resource_count = 1
    resource_group_indices = [[0]]
    result = task_allocator.allocate_task(
        resource_windows=resource_windows,
        resource_ids=resource_ids,
        task_duration=task_duration,
        resource_count=resource_count,
        resource_group_indices=resource_group_indices,
    )
    assert result is None


def test_allocate_task_with_invalid_input(task_allocator):
    resource_windows_dict = {1: np.array([[0, 1, 1, 0], [2, 3, 1, -1]])}
    task_duration = "invalid"
    resource_count = 1
    resource_group_indices = [[0]]
    with pytest.raises(TypeError):
        task_allocator.allocate_task(
            resource_windows_dict, task_duration, resource_count, resource_group_indices
        )


def test_allocate_task_with_empty_input(task_allocator):
    resource_windows_dict = {}
    task_duration = 1
    resource_count = 1
    resource_group_indices = [[]]
    with pytest.raises(Exception):  # or the specific exception that you expect
        task_allocator.allocate_task(
            resource_windows_dict, task_duration, resource_count, resource_group_indices
        )


# def test_allocate_task_with_negative_task_duration(task_allocator):
#     resource_windows_dict = {1: np.array([[0, 1, 1, 0], [2, 3, 1, -1]])}
#     task_duration = -1
#     resource_count = 1
#     resource_group_indices = [[0]]
#     with pytest.raises(Exception):  # or your expected behavior
#         task_allocator.allocate_task(
#             resource_windows_dict, task_duration, resource_count, resource_group_indices
#         )


def test_fill_array_except_largest_group_per_row(task_allocator):
    array = np.array(
        [
            [5, 0, 10],
            [10, 30, 20],
            [1, 1, 1],
        ]
    )
    group_indices = [[0, 1], [2]]
    result = task_allocator._fill_array_except_largest_group_per_row(
        array, group_indices
    )
    expected = np.array(
        [
            [0, 0, 10],
            [10, 30, 0],
            [1, 1, 0],
        ]
    )
    assert np.array_equal(result, expected)


def test_expand_array(task_allocator):
    arr = np.array(
        [
            [5, 0],
            [10, 10],
        ]
    )
    new_boundaries = np.array([7, 12])
    result = task_allocator._expand_array(arr, new_boundaries)
    expected = np.array(
        [
            [5, 0],
            [7, 4],
            [10, 10],
            [12, 0],
        ]
    )
    assert np.array_equal(result, expected)


test_case_values = [
    # trim both intervals
    (
        "No Resets",
        [
            np.array([[0, 10, 10, 0], [20, 30, 10, 0]]),
            np.array([[5, 15, 10, 0], [20, 30, 10, 0]]),
        ],
        np.array(
            [
                [0, 0, 0],
                [5, 5, 0],
                [10, 10, 5],
                [15, 10, 10],
                [20, 10, 10],
                [30, 20, 20],
            ]
        ),
    ),
    (
        "With resets and high durations",
        [
            np.array([[0, 10, 20, 0], [20, 30, 10, -1]]),
            np.array([[5, 15, 10, 0], [20, 30, 10, -1]]),
            np.array([[0, 30, 60, 0]]),
        ],
        np.array(
            [
                [0, 0, 0, 0],
                [5, 10, 0, 10],
                [10, 20, 5, 20],
                [15, 20, 10, 30],
                [20, 0, 0, 40],
                [30, 10, 10, 60],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "test_name, resource_windows_dict, expected",
    test_case_values,
    ids=[case[0] for case in test_case_values],
)
def test_create_matrix(test_name, task_allocator, resource_windows_dict, expected):
    result = task_allocator.create_matrix(resource_windows_dict)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([0, 1, 3, 0, 1, 1]), 3),
        (np.array([0, 2, 3, 4, 5]), 0),
        (np.array([2, 3, 4, 5, 6]), 0),
    ],
)
def test_get_window_start_index(task_allocator, array, expected):
    result = task_allocator._get_window_start_index(array)
    assert np.array_equal(result, expected)


def test_mask_smallest_except_k_largest(task_allocator):
    array = np.array(
        [
            [5, 0, 10],
            [10, 30, 20],
            [1, 1, 1],
        ]
    )
    result = task_allocator._mask_smallest_except_k_largest(array, 2).mask
    expected = np.array(
        [
            [False, True, False],
            [True, False, False],
            [True, False, False],
        ]
    )
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([0, 1, 5, -1, 10]), [0, 1, 6, 0, 10]),
        (np.array([-1, 2, 3, 0, 4]), [0, 2, 5, 5, 9]),
        (np.array([0, -1, 2, 4, -1]), [0, 0, 2, 6, 0]),
    ],
)
def test_cumsum_reset_at_minus_one(task_allocator, array, expected):
    result = task_allocator._cumsum_reset_at_minus_one(array)
    assert np.array_equal(result, expected)
