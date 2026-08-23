"""Compatibility helpers for supported nvMolKit clustering result shapes."""

import numpy as np


def normalize_fused_butina_result(raw_result, *, molecule_count):
    """Normalize supported fused-Butina results and validate their partition."""

    def _integer_vector(value, *, require_vector=True):
        numpy_method = getattr(value, "numpy", None)
        if numpy_method is not None:
            value = numpy_method()
        if not require_vector:
            return value

        if isinstance(value, (list, tuple)) and any(
            isinstance(item, (bool, np.bool_)) for item in value
        ):
            raise ValueError
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype.kind not in "iuf":
            raise ValueError
        if array.dtype.kind == "f" and (
            not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all()
        ):
            raise ValueError

        with np.errstate(invalid="ignore", over="ignore"):
            integers = array.astype(int, copy=True)
        if not np.equal(array, integers).all():
            raise ValueError
        return integers

    try:
        if type(molecule_count) is not int or molecule_count < 1:
            raise ValueError

        if not isinstance(raw_result, (tuple, list)):
            raise ValueError
        result_length = len(raw_result)
        if result_length not in (2, 3):
            raise ValueError

        if result_length == 2:
            labels = _integer_vector(raw_result[0])
            centroids = _integer_vector(raw_result[1])
            if len(labels) != molecule_count or np.any(labels < 0):
                raise ValueError

            cluster_ids = np.unique(labels)
            if not np.array_equal(cluster_ids, np.arange(len(cluster_ids), dtype=int)):
                raise ValueError
            clusters = tuple(
                tuple(np.flatnonzero(labels == cluster_id).tolist())
                for cluster_id in cluster_ids
            )
        else:
            member_lists = _integer_vector(raw_result[0], require_vector=False)
            member_vectors = tuple(_integer_vector(members) for members in member_lists)
            clusters = tuple(
                tuple(int(member) for member in members) for members in member_vectors
            )
            cumulative_sizes = _integer_vector(raw_result[1])
            centroids = _integer_vector(raw_result[2])

            if not clusters or any(not cluster for cluster in clusters):
                raise ValueError
            expected_cumulative_sizes = np.concatenate(
                (
                    np.zeros(1, dtype=int),
                    np.cumsum(
                        np.asarray([len(cluster) for cluster in clusters], dtype=int)
                    ),
                )
            )
            if not np.array_equal(cumulative_sizes, expected_cumulative_sizes):
                raise ValueError

            assigned = [member for cluster in clusters for member in cluster]
            if len(assigned) != molecule_count or sorted(assigned) != list(
                range(molecule_count)
            ):
                raise ValueError

            labels = np.full(molecule_count, -1, dtype=int)
            for cluster_id, cluster in enumerate(clusters):
                labels[list(cluster)] = cluster_id

        if len(centroids) != len(clusters):
            raise ValueError
        if any(
            int(centroid) not in clusters[cluster_id]
            for cluster_id, centroid in enumerate(centroids)
        ):
            raise ValueError

        if len(labels) != molecule_count or np.any(labels < 0):
            raise ValueError
        if not np.array_equal(np.unique(labels), np.arange(len(clusters), dtype=int)):
            raise ValueError

        return labels, clusters, centroids
    except Exception:
        pass

    raise ValueError("Malformed fused Butina result.")
