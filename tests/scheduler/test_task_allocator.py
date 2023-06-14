import numpy as np
import pytest

from pyplanpro.scheduler.heuristic_solver.task_allocator import TaskAllocator


@pytest.fixture
def task_allocator():
    return TaskAllocator()


@pytest.mark.parametrize(
    "task_duration, resource_count, expected",
    [
        (10, 1, {1: (10, 20)}),
        (15, 2, {1: (10, 20), 2: (15, 20)}),
        (20, 2, {1: (10, 20), 2: (15, 25)}),
    ],
)
def test_find_earliest_solution(
    task_allocator, task_duration, resource_count, expected
):
    resource_windows_dict = {
        0: [[(0, 5), (30, 35)]],
        1: [[(10, 20)]],
        2: [[(15, 30)]],
    }
    result = task_allocator.find_earliest_solution(
        task_duration, resource_windows_dict, resource_count
    )
    assert result == expected


def test_solution_to_resource_windows(task_allocator):
    solution_matrix = np.array(
        [
            [0, 0, 0],
            [2, 2, 0],
            [3, 3, 1],
            [4, 4, 2],
            [5, 5, 3],
        ]
    )
    resources = [2, 1, 4]
    task_duration = 5
    result = task_allocator._solution_to_resource_windows(
        solution_matrix, task_duration
    )
    expected = {[(0, 4), (0, 4)]}
    assert np.array_equal(result, expected)


def test_create_matrix(task_allocator):
    resource_windows_dict = {
        0: [[(0, 5), (9, 10)]],
        1: [[(6, 12)], [(13, 15)]],
    }
    result = task_allocator.create_matrix(resource_windows_dict)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [6.0, 5.0, 0.0],
            [9.0, 5.0, 3.0],
            [10.0, 6.0, 4.0],
            [12.0, 6.0, 6.0],
            [13.0, 6.0, 0.0],
            [15.0, 6.0, 2.0],
        ]
    )
    assert np.array_equal(result, expected)


def test_fill_window(task_allocator):
    matrix = np.array(
        [
            [0, 0],
            [10, 0],
            [15, 0],
            [30, 0],
            [35, 0],
        ]
    )
    window = [(10.0, 30.0)]
    task_allocator._fill_window(matrix, window, 1)
    expected = np.array(
        [
            [0, 0],
            [10, -1],
            [15, 5],
            [30, 15],
            [35, 0],
        ]
    )
    assert np.array_equal(matrix, expected)


def test_distribute(task_allocator):
    total = 30
    timepoints = [10, 20, 25]
    result = task_allocator._distribute(total, timepoints)
    expected = [0, 20, 10]
    assert np.sum(expected) == total
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([0, 1, 2, 3, 4, 4, 4]), 4),
        (np.array([0, 1, 1, 2, 3, 4, 4]), 1),
        (np.array([2, 2, 2, 2]), 0),
    ],
)
def test_window_end_index(task_allocator, array, expected):
    result = task_allocator._window_end_index(array)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([0, 1, 3, 0, 1, 1]), 3),
        (np.array([0, 2, 3, 4, 5]), 0),
        (np.array([2, 3, 4, 5, 6]), 0),
    ],
)
def test_window_start_index(task_allocator, array, expected):
    result = task_allocator._window_start_index(array)
    assert np.array_equal(result, expected)


def test_pad_array_with_zeros(task_allocator):
    array = np.array([0, 10, 15, 30])
    result = task_allocator._pad_array_with_zeros(array, 2)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )
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


def test_linear_expand_matrix(task_allocator):
    array = np.array(
        [
            [5, 0, 10],
            [10, 30, 20],
        ]
    )
    result = task_allocator._linear_expand_matrix(array, 1)
    expected = np.array(
        [
            [5.0, 0.0, 10.0],
            [6.0, 6.0, 12.0],
            [7.0, 12.0, 14.0],
            [8.0, 18.0, 16.0],
            [9.0, 24.0, 18.0],
            [10.0, 30.0, 20.0],
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
