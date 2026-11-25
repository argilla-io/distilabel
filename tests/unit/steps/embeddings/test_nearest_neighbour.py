# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import faiss
import numpy as np
import pytest
from datasets import Dataset

from distilabel.steps.embeddings.nearest_neighbour import FaissNearestNeighbour


class TestFaissNearestNeighbour:
    def test_process(self) -> None:
        step = FaissNearestNeighbour()

        step.load()

        results = next(
            step.process(
                inputs=[
                    {"embedding": [0.1, -0.4, 0.7, 0.2]},
                    {"embedding": [-0.3, 0.9, 0.1, -0.5]},
                    {"embedding": [0.6, 0.2, -0.1, 0.8]},
                    {"embedding": [-0.2, -0.6, 0.4, 0.3]},
                    {"embedding": [0.9, 0.1, -0.3, -0.2]},
                    {"embedding": [0.4, -0.7, 0.6, 0.1]},
                    {"embedding": [-0.5, 0.3, -0.2, 0.9]},
                    {"embedding": [0.7, 0.5, -0.4, -0.1]},
                    {"embedding": [-0.1, -0.9, 0.8, 0.6]},
                ]
            )
        )

        expected = [
            {
                "embedding": [0.1, -0.4, 0.7, 0.2],
                "nn_indices": [5],
                "nn_scores": [0.19999998807907104],
            },
            {
                "embedding": [-0.3, 0.9, 0.1, -0.5],
                "nn_indices": [7],
                "nn_scores": [1.5699999332427979],
            },
            {
                "embedding": [0.6, 0.2, -0.1, 0.8],
                "nn_indices": [7],
                "nn_scores": [1.0000001192092896],
            },
            {
                "embedding": [-0.2, -0.6, 0.4, 0.3],
                "nn_indices": [0],
                "nn_scores": [0.23000000417232513],
            },
            {
                "embedding": [0.9, 0.1, -0.3, -0.2],
                "nn_indices": [7],
                "nn_scores": [0.2200000137090683],
            },
            {
                "embedding": [0.4, -0.7, 0.6, 0.1],
                "nn_indices": [0],
                "nn_scores": [0.19999998807907104],
            },
            {
                "embedding": [-0.5, 0.3, -0.2, 0.9],
                "nn_indices": [2],
                "nn_scores": [1.2400000095367432],
            },
            {
                "embedding": [0.7, 0.5, -0.4, -0.1],
                "nn_indices": [4],
                "nn_scores": [0.2200000137090683],
            },
            {
                "embedding": [-0.1, -0.9, 0.8, 0.6],
                "nn_indices": [3],
                "nn_scores": [0.3499999940395355],
            },
        ]

        assert len(results) == len(expected)
        for actual, reference in zip(results, expected):
            assert actual["embedding"] == reference["embedding"]
            assert actual["nn_indices"] == reference["nn_indices"]
            assert actual["nn_scores"] == pytest.approx(reference["nn_scores"])


@pytest.mark.parametrize("batch_size", [1, 2, 10])
@pytest.mark.parametrize("metric", [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT])
def test_search_excludes_actual_self_with_ties(batch_size, metric):
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    dataset = Dataset.from_dict({"embedding": embeddings.tolist()})
    dataset.add_faiss_index(column="embedding", metric_type=metric)
    step = FaissNearestNeighbour(k=2, search_batch_size=batch_size)
    result = step._search(dataset)
    for row_id, row in enumerate(result):
        assert row_id not in row["nn_indices"]
        assert len(row["nn_indices"]) == 2
        assert len(set(row["nn_indices"])) == 2
        for index, score in zip(row["nn_indices"], row["nn_scores"]):
            expected = (
                float(embeddings[row_id] @ embeddings[index])
                if metric == faiss.METRIC_INNER_PRODUCT
                else float(np.square(embeddings[row_id] - embeddings[index]).sum())
            )
            assert score == pytest.approx(expected)
        if row_id < 3:
            assert set(row["nn_indices"]) == {0, 1, 2} - {row_id}


@pytest.mark.parametrize("batch_size", [1, 3])
def test_search_keeps_best_neighbor_when_self_is_not_first(batch_size):
    dataset = Dataset.from_dict({"embedding": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]})
    dataset.add_faiss_index(column="embedding", metric_type=faiss.METRIC_INNER_PRODUCT)
    result = FaissNearestNeighbour(k=1, search_batch_size=batch_size)._search(dataset)
    assert result["nn_indices"] == [[2], [2], [1]]
    assert result["nn_scores"] == [[3.0], [6.0], [6.0]]


@pytest.mark.parametrize("count,k", [(1, 1), (2, 5), (3, 3)])
def test_search_ignores_missing_neighbor_sentinels(count, k):
    dataset = Dataset.from_dict({"embedding": [[float(i), 0.0] for i in range(count)]})
    dataset.add_faiss_index(column="embedding")
    result = FaissNearestNeighbour(k=k, search_batch_size=1)._search(dataset)
    for row_id, row in enumerate(result):
        assert set(row["nn_indices"]) == set(range(count)) - {row_id}
        assert len(row["nn_scores"]) == count - 1
