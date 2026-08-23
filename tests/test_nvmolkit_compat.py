import ast
import functools
import importlib.util
import inspect
import textwrap
from collections import deque
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "notebooks" / "nvmolkit_compat.py"
MALFORMED_MESSAGE = "Malformed fused Butina result."


class FakeAsyncResult:
    def __init__(self, value):
        self.value = value
        self.numpy_calls = 0

    def numpy(self):
        self.numpy_calls += 1
        return self.value


class FailingAsyncResult:
    def numpy(self):
        raise RuntimeError("sensitive-token")


@functools.cache
def _load_nvmolkit_compat():
    if not MODULE_PATH.is_file():
        pytest.fail("The nvMolKit compatibility module has not been created.")
    spec = importlib.util.spec_from_file_location(
        "nvmolkit_compat_for_tests", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize(raw_result, *, molecule_count):
    return _load_nvmolkit_compat().normalize_fused_butina_result(
        raw_result, molecule_count=molecule_count
    )


def _assert_integer_array(actual, expected):
    assert isinstance(actual, np.ndarray)
    assert actual.ndim == 1
    assert np.issubdtype(actual.dtype, np.integer)
    np.testing.assert_array_equal(actual, expected)


def _assert_malformed(raw_result, *, molecule_count=4):
    with pytest.raises(ValueError) as captured:
        _normalize(raw_result, molecule_count=molecule_count)
    assert str(captured.value) == MALFORMED_MESSAGE
    assert "sensitive-token" not in str(captured.value)


def test_normalizes_v05_member_lists_and_async_vectors():
    first_cluster = FakeAsyncResult(np.array([2.0, 0.0]))
    second_cluster = FakeAsyncResult(np.array([3]))
    third_cluster = FakeAsyncResult([1, 4])
    cumulative_sizes = FakeAsyncResult(np.array([0.0, 2.0, 3.0, 5.0]))
    centroids = FakeAsyncResult(np.array([2.0, 3.0, 1.0]))

    labels, clusters, normalized_centroids = _normalize(
        (
            (first_cluster, second_cluster, third_cluster),
            cumulative_sizes,
            centroids,
        ),
        molecule_count=5,
    )

    _assert_integer_array(labels, [0, 2, 0, 1, 2])
    assert clusters == ((2, 0), (3,), (1, 4))
    _assert_integer_array(normalized_centroids, [2, 3, 1])
    assert first_cluster.numpy_calls == 1
    assert second_cluster.numpy_calls == 1
    assert third_cluster.numpy_calls == 1
    assert cumulative_sizes.numpy_calls == 1
    assert centroids.numpy_calls == 1


def test_normalizes_v06_async_cluster_ids():
    cluster_ids = FakeAsyncResult(np.array([1.0, 0.0, 1.0, 2.0, 0.0]))
    centroids = FakeAsyncResult(np.array([4.0, 0.0, 3.0]))

    labels, clusters, normalized_centroids = _normalize(
        [cluster_ids, centroids], molecule_count=5
    )

    _assert_integer_array(labels, [1, 0, 1, 2, 0])
    assert clusters == ((1, 4), (0, 2), (3,))
    _assert_integer_array(normalized_centroids, [4, 0, 3])
    assert cluster_ids.numpy_calls == 1
    assert centroids.numpy_calls == 1


def test_normalizer_source_is_self_contained_with_one_nested_converter():
    normalizer = _load_nvmolkit_compat().normalize_fused_butina_result
    source = textwrap.dedent(inspect.getsource(normalizer))
    function_node = ast.parse(source).body[0]
    nested_functions = [
        node for node in function_node.body if isinstance(node, ast.FunctionDef)
    ]
    assert len(nested_functions) == 1

    isolated_globals = {"np": np}
    exec(source, isolated_globals)
    isolated_normalizer = isolated_globals["normalize_fused_butina_result"]
    labels, clusters, centroids = isolated_normalizer(
        ([0, 1, 0], [0, 1]), molecule_count=3
    )
    _assert_integer_array(labels, [0, 1, 0])
    assert clusters == ((0, 2), (1,))
    _assert_integer_array(centroids, [0, 1])

    labels, clusters, centroids = isolated_normalizer(
        (([2, 0], [1]), [0, 2, 3], [2, 1]), molecule_count=3
    )
    _assert_integer_array(labels, [0, 1, 0])
    assert clusters == ((2, 0), (1,))
    _assert_integer_array(centroids, [2, 1])


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "molecule_count",
    [None, True, False, 0, -1, 1.0, "4", np.int64(4), IntSubclass(4)],
)
def test_rejects_a_molecule_count_that_is_not_a_positive_builtin_int(
    molecule_count,
):
    _assert_malformed(([0, 1, 0, 1], [0, 1]), molecule_count=molecule_count)


@pytest.mark.parametrize(
    "raw_result",
    [
        None,
        (),
        ([0, 1, 0, 1],),
        ([0, 1, 0, 1], [0, 1], [0, 1], [0, 1]),
        FailingAsyncResult(),
    ],
)
def test_rejects_results_without_exactly_two_or_three_fields(raw_result):
    _assert_malformed(raw_result)


@pytest.mark.parametrize(
    "raw_result",
    [
        np.array([[0, 1], [0, 1]]),
        {0: [0, 1], 1: [0, 1]},
        "ab",
        deque(([0, 1], [0, 1])),
    ],
)
def test_rejects_non_list_or_tuple_result_containers(raw_result):
    _assert_malformed(raw_result, molecule_count=2)


@pytest.mark.parametrize(
    "cluster_ids",
    [
        [[0, 1], [0, 1]],
        [0, 1.5, 0, 1],
        [0, "sensitive-token", 0, 1],
        [0, np.nan, 0, 1],
        [0, np.inf, 0, 1],
        np.array([True, False, True, False]),
    ],
)
def test_rejects_v06_cluster_ids_that_are_not_one_dimensional_integer_values(
    cluster_ids,
):
    _assert_malformed((cluster_ids, [0, 1]))


@pytest.mark.parametrize(
    "centroids",
    [
        [[0, 1]],
        [0.5, 1],
        ["sensitive-token", 1],
        [np.nan, 1],
        [np.inf, 1],
        np.array([True, False]),
    ],
)
def test_rejects_v06_centroids_that_are_not_one_dimensional_integer_values(
    centroids,
):
    _assert_malformed(([0, 1, 0, 1], centroids))


@pytest.mark.parametrize(
    "cluster_ids",
    [[0, 1, 0], [0, 1, 0, 1, 0]],
)
def test_rejects_v06_label_vectors_with_the_wrong_length(cluster_ids):
    _assert_malformed((cluster_ids, [0, 1]))


@pytest.mark.parametrize(
    "cluster_ids,centroids",
    [
        ([0, -1, 0, 1], [0, 3]),
        ([0, 2, 0, 2], [0, 1]),
        ([1, 1, 1, 1], [0]),
    ],
)
def test_rejects_v06_labels_that_are_negative_or_not_contiguous(cluster_ids, centroids):
    _assert_malformed((cluster_ids, centroids))


@pytest.mark.parametrize(
    "member_lists,cumulative_sizes,centroids",
    [
        (([], [0, 1, 2, 3]), [0, 0, 4], [0, 1]),
        (([0, 2], [1, 2]), [0, 2, 4], [0, 1]),
        (([0, 2], [1]), [0, 2, 3], [0, 1]),
        (([0, 2], [1, 4]), [0, 2, 4], [0, 1]),
        (([0, 2], [-1, 1, 3]), [0, 2, 5], [0, 1]),
    ],
)
def test_rejects_v05_clusters_that_are_not_an_exact_nonempty_partition(
    member_lists, cumulative_sizes, centroids
):
    _assert_malformed((member_lists, cumulative_sizes, centroids))


@pytest.mark.parametrize(
    "member_lists",
    [
        ([[0, 2]], [1, 3]),
        ([0.5, 2], [1, 3]),
        (["sensitive-token", 2], [1, 3]),
        ([True, False], [1, 3]),
    ],
)
def test_rejects_v05_member_lists_that_are_not_1d_integer_values(member_lists):
    _assert_malformed((member_lists, [0, 2, 4], [0, 1]))


@pytest.mark.parametrize(
    "cumulative_sizes",
    [
        [[0, 2, 4]],
        [0, 2.5, 4],
        [0, "sensitive-token", 4],
        [False, 2, 4],
        [2],
        [2, 4],
        [0, 2],
        [0, 2, 3],
        [0, 2, 5],
    ],
)
def test_rejects_invalid_v05_cumulative_sizes(cumulative_sizes):
    _assert_malformed((([0, 2], [1, 3]), cumulative_sizes, [0, 1]))


@pytest.mark.parametrize(
    "centroids",
    [
        [[0, 1]],
        [0.5, 1],
        ["sensitive-token", 1],
        [True, False],
    ],
)
def test_rejects_v05_centroids_that_are_not_1d_integer_values(centroids):
    _assert_malformed((([0, 2], [1, 3]), [0, 2, 4], centroids))


@pytest.mark.parametrize(
    "raw_result",
    [
        ([0, 1, 0, 1], [0]),
        (([0, 2], [1, 3]), [0, 2, 4], [0]),
    ],
)
def test_rejects_a_centroid_count_that_does_not_match_the_cluster_count(
    raw_result,
):
    _assert_malformed(raw_result)


@pytest.mark.parametrize(
    "raw_result",
    [
        ([0, 1, 0, 1], [1, 0]),
        (([0, 2], [1, 3]), [0, 2, 4], [1, 0]),
    ],
)
def test_rejects_a_centroid_outside_its_corresponding_cluster(raw_result):
    _assert_malformed(raw_result)


@pytest.mark.parametrize(
    "raw_result",
    [
        (FailingAsyncResult(), [0, 1]),
        ([0, 1, 0, 1], FailingAsyncResult()),
        ((FailingAsyncResult(), [1, 3]), [0, 2, 4], [0, 1]),
        (([0, 2], [1, 3]), FailingAsyncResult(), [0, 1]),
    ],
)
def test_wraps_async_conversion_failures_without_disclosing_details(raw_result):
    _assert_malformed(raw_result)


def test_async_conversion_failure_drops_the_raw_exception_chain():
    with pytest.raises(ValueError) as captured:
        _normalize((FailingAsyncResult(), [0, 1]), molecule_count=4)

    assert str(captured.value) == MALFORMED_MESSAGE
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None
