"""Cold-process native numerics check, without the duplicate-OpenMP bypass."""

import json
import os
import sys
from typing import Any, Dict

from utils.native_runtime import _loaded_openmp_paths, prepare_cli_native_runtime, require_native_runtime


def main() -> None:
    prepare_cli_native_runtime()
    require_native_runtime()
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() != "FALSE":
        raise RuntimeError("Start this check with KMP_DUPLICATE_LIB_OK=FALSE")

    import faiss
    import numpy as np
    import torch
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(128, 8)).astype("float32")
    for _ in range(3):
        tensor = torch.from_numpy(vectors.copy())
        actual = (tensor @ tensor.T).numpy()
        np.testing.assert_allclose(actual, vectors @ vectors.T, rtol=1e-5, atol=1e-5)
        index = faiss.IndexFlatL2(8)
        index.add(vectors)
        distances, indices = index.search(vectors[:4], 1)
        assert indices[:, 0].tolist() == [0, 1, 2, 3]
        np.testing.assert_allclose(distances[:, 0], np.zeros(4), atol=1e-5)
        samples = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
        centers = KMeans(n_clusters=2, n_init=1, random_state=42).fit(samples).cluster_centers_
        np.testing.assert_allclose(centers[np.argsort(centers[:, 0])], [[0.0, 0.5], [10.0, 10.5]])
    require_native_runtime()
    assert os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() == "FALSE"
    evidence: Dict[str, Any] = {"platform": sys.platform, "rounds": 3, "native_numerics": "passed", "KMP": "FALSE"}
    if sys.platform == "darwin":
        images = _loaded_openmp_paths()
        assert len(images) == 1
        evidence["openmp_images"] = sorted(str(path) for path in images)
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
