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

from distilabel.pipeline.local import Pipeline
from distilabel.steps.deita import DeitaFiltering


class TestDeitaFiltering:
    def test_process(self) -> None:
        deita_filtering = DeitaFiltering(
            name="deita_filtering",
            data_budget=1,
            pipeline=Pipeline(name="unit-test"),
        )

        deita_filtering.load()

        result = next(
            deita_filtering.process(
                [
                    {
                        "evol_instruction_score": 0.5,
                        "evol_response_score": 0.5,
                        "embedding": [-8.12729941, -5.24642847, -6.34003029],
                    },
                    {
                        "evol_instruction_score": 0.6,
                        "evol_response_score": 0.6,
                        "embedding": [2.99329242, 0.7800932, 0.7799726],
                    },
                    {
                        "evol_instruction_score": 0.7,
                        "evol_response_score": 0.7,
                        "embedding": [10.29041806, 14.33088073, 13.00557506],
                    },
                ]
            )
        )

        assert result == [
            {
                "evol_instruction_score": 0.5,
                "evol_response_score": 0.5,
                "embedding": [-8.12729941, -5.24642847, -6.34003029],
                "deita_score": 0.25,
                "deita_score_computed_with": [
                    "evol_instruction_score",
                    "evol_response_score",
                ],
                "nearest_neighbor_distance": 1.9042812683723933,
            }
        ]


@pytest.mark.parametrize("count", [1, 2, 31, 32, 33, 65])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_manhattan_matches_full_tensor(count, dtype):
    embeddings = np.random.default_rng(42).normal(size=(count, 17)).astype(dtype)
    if count > 2:
        embeddings[-1] = embeddings[0]
    distances = np.abs(embeddings[:, None] - embeddings).sum(-1)
    np.fill_diagonal(distances, np.inf)
    expected = distances.min(axis=1)
    result = DeitaFiltering(data_budget=1)._manhattan_distance(embeddings)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_manhattan_preserves_nonfinite_propagation():
    embeddings = np.zeros((65, 3))
    embeddings[-1, 0] = np.nan
    distances = np.abs(embeddings[:, None] - embeddings).sum(-1)
    np.fill_diagonal(distances, np.inf)
    np.testing.assert_array_equal(
        DeitaFiltering(data_budget=1)._manhattan_distance(embeddings),
        distances.min(axis=1),
    )


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("budget", [1, 20, 65])
def test_manhattan_selection_matches_full_tensor(normalize, budget):
    embeddings = np.random.default_rng(123).normal(size=(65, 17))
    embeddings[-1] = embeddings[0]
    measured = embeddings.copy()
    if normalize:
        measured /= np.linalg.norm(measured, axis=1, keepdims=True)
    distances = np.abs(measured[:, None] - measured).sum(-1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    threshold = float(np.median(nearest))
    rows = [
        {
            "id": i,
            "embedding": embedding.tolist(),
            "evol_instruction_score": float(i + 1),
            "evol_response_score": 0.5,
        }
        for i, embedding in enumerate(embeddings)
    ]
    expected_ids = [i for i in reversed(range(65)) if nearest[i] >= threshold][:budget]
    step = DeitaFiltering(
        data_budget=budget,
        distance_metric="manhattan",
        normalize_embeddings=normalize,
        diversity_threshold=threshold,
        pipeline=Pipeline(name="manhattan-test"),
    )
    step.load()
    result = next(step.process(rows))
    assert [row["id"] for row in result] == expected_ids
    for row in result:
        assert row["nearest_neighbor_distance"] == nearest[row["id"]]
        assert row["deita_score"] == (row["id"] + 1) * 0.5
