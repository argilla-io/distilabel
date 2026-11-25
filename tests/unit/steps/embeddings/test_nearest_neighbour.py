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

import numpy as np
import pytest
from datasets import Dataset

from distilabel.steps.embeddings.nearest_neighbour import FaissNearestNeighbour


class TestFaissNearestNeighbour:
    @pytest.mark.parametrize("batch_size", [1, 3, 16])
    @pytest.mark.parametrize("k", [1, 12])
    @pytest.mark.parametrize("existing_outputs", [False, True])
    @pytest.mark.parametrize("indexed_view", [False, True])
    @pytest.mark.parametrize("metric_type", [0, 1])
    def test_search_preserves_payload(
        self, batch_size, k, existing_outputs, indexed_view, metric_type
    ) -> None:
        data = {
            "text": [None, "", "hello", "long text" * 100, "unicode: \u00e9"],
            "embedding": np.random.default_rng(42).normal(size=(5, 4)).tolist(),
            "metadata": [{"id": i, "tags": ["a", "b"]} for i in range(5)],
        }
        if existing_outputs:
            # Existing columns can occur before or between the input columns.
            data = {"nn_scores": [[0.0]] * 5, **data, "nn_indices": [[0]] * 5}
        dataset = Dataset.from_dict(data)
        order = [4, 1, 3, 0, 2] if indexed_view else list(range(5))
        if indexed_view:
            dataset = dataset.select(order)
        dataset.add_faiss_index("embedding", metric_type=metric_type)
        step = FaissNearestNeighbour(k=k, search_batch_size=batch_size)

        # The previous full-row map is the oracle, using the same real index.
        def full_row_search(examples):
            results = dataset.search_batch(
                "embedding", np.array(examples["embedding"]), k=k + 1
            )
            examples["nn_indices"] = [x[1:] for x in results.total_indices]
            examples["nn_scores"] = [x[1:] for x in results.total_scores]
            return examples

        expected = dataset.map(full_row_search, batched=True, batch_size=batch_size)
        actual = step._search(dataset)
        assert actual.to_list() == expected.to_list()
        assert actual.features == expected.features
        assert actual.column_names == expected.column_names
        assert dataset.to_dict() == {
            key: [value[i] for i in order] for key, value in data.items()
        }
        # Wide payload buffers must be reused, not serialized into a second table.
        if not indexed_view:
            for column in ("text", "embedding", "metadata"):
                assert [
                    b.address if b is not None else None
                    for b in actual.data.column(column).chunks[0].buffers()
                ] == [
                    b.address if b is not None else None
                    for b in dataset.data.column(column).chunks[0].buffers()
                ]
        assert dataset.list_indexes() == ["embedding"]

    def test_search_empty(self) -> None:
        dataset = Dataset.from_dict({"embedding": [], "text": []})
        assert FaissNearestNeighbour()._search(dataset) is dataset

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

        assert results == [
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
