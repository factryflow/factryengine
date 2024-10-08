import numpy as np
import pytest
from factryengine.scheduler.heuristic_solver.matrix import Matrix


@pytest.fixture
def matrix_data_dict():
    return {
        "resource_ids": np.array([1, 2, 3]),
        "intervals": np.array([0, 1, 2]),
        "resource_matrix": np.ma.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 1, 0],
            ]
        ),
    }


def test_can_create_matrix(matrix_data_dict):
    matrix = Matrix(**matrix_data_dict)

    assert np.array_equal(matrix.resource_ids, matrix_data_dict["resource_ids"])
    assert np.array_equal(matrix.intervals, matrix_data_dict["intervals"])
    assert np.array_equal(matrix.resource_matrix, matrix_data_dict["resource_matrix"])


def test_can_merge_matrix(matrix_data_dict):
    matrix1 = Matrix(**matrix_data_dict)
    matrix2 = Matrix(**matrix_data_dict)
    matrix2_resource_ids = np.array([4, 5, 6])
    matrix2.resource_ids = matrix2_resource_ids

    merged_matrix = Matrix.merge([matrix1, matrix2])

    expected_resource_ids = np.concatenate([matrix1.resource_ids, matrix2.resource_ids])
    expected_resource_matrix = np.ma.hstack(
        [matrix1.resource_matrix, matrix2.resource_matrix]
    )

    assert np.array_equal(merged_matrix.resource_ids, expected_resource_ids)
    assert np.array_equal(merged_matrix.resource_matrix, expected_resource_matrix)
    assert np.array_equal(merged_matrix.intervals, matrix_data_dict["intervals"])


def test_get_value_error_on_merge_with_different_intervals(matrix_data_dict):
    matrix1 = Matrix(**matrix_data_dict)
    matrix2 = Matrix(**matrix_data_dict)
    matrix2.intervals = np.array([0, 1, 2, 3])

    with pytest.raises(ValueError):
        Matrix.merge([matrix1, matrix2])


def test_can_compare_update_mask_and_merge(matrix_data_dict):
    matrix1 = Matrix(**matrix_data_dict)
    matrix2 = Matrix(**matrix_data_dict)

    matrix1.resource_matrix = np.ma.array(
        [
            [0, 0, 0],
            [10, 10, 10],
            [1, 1, 1],
        ]
    )
    matrix2.resource_matrix = np.ma.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [10, 10, 10],
        ]
    )

    merged_matrix = Matrix.compare_update_mask_and_merge([matrix1, matrix2])

    expected_mask = np.array(
        [
            [False, False, False, True, True, True],  # keep the first row of matrix1
            [False, False, False, True, True, True],
            [True, True, True, False, False, False],
        ]
    )
    assert np.array_equal(merged_matrix.resource_matrix.mask, expected_mask)


def test_can_trim_matrix(matrix_data_dict):
    matrix1 = Matrix(**matrix_data_dict)
    matrix2 = Matrix(**matrix_data_dict)
    matrix2.intervals = np.array([0, 1])

    trimmed_matrix = Matrix.trim_end(matrix1, matrix2)

    assert np.array_equal(trimmed_matrix.intervals, matrix2.intervals)
    assert np.array_equal(trimmed_matrix.resource_ids, matrix1.resource_ids)
    assert np.array_equal(
        trimmed_matrix.resource_matrix,
        matrix1.resource_matrix[: len(matrix2.intervals)],
    )


def test_get_value_error_on_trim_with_different_intervals(matrix_data_dict):
    matrix1 = Matrix(**matrix_data_dict)
    matrix2 = Matrix(**matrix_data_dict)
    matrix2.intervals = np.array([0, 1, 2, 3])

    with pytest.raises(ValueError):
        Matrix.trim_end(matrix1, matrix2)
