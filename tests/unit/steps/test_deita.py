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

import copy

import numpy as np
import pytest

from distilabel.pipeline.local import Pipeline
from distilabel.steps.deita import DeitaFiltering


class TestDeitaFiltering:
    @staticmethod
    def _full_cosine_distance(embeddings):
        distances = 1 - np.dot(embeddings, embeddings.T)
        np.fill_diagonal(distances, np.inf)
        return distances.min(axis=1)

    @pytest.mark.parametrize("rows", [1, 511, 512, 513, 1025])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_cosine_distance_tiles(self, rows, dtype, normalize):
        step = DeitaFiltering(data_budget=1, pipeline=Pipeline(name="cosine-tiles"))
        embeddings = np.random.default_rng(7).normal(size=(rows, 33)).astype(dtype)
        if normalize:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        if rows > 512:
            embeddings[512] = embeddings[0]
        original = embeddings.copy()
        expected = self._full_cosine_distance(embeddings)
        actual = step._cosine_distance(embeddings)
        assert actual.dtype == expected.dtype
        tolerance = (1e-6 if normalize else 1e-5) if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
        np.testing.assert_array_equal(embeddings, original)

    def test_cosine_distance_noncontiguous(self):
        step = DeitaFiltering(data_budget=1, pipeline=Pipeline(name="cosine-strides"))
        embeddings = np.random.default_rng(9).normal(size=(1026, 64))[::2, ::2]
        np.testing.assert_allclose(
            step._cosine_distance(embeddings),
            self._full_cosine_distance(embeddings),
            rtol=0,
            atol=1e-12,
        )

    @pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
    def test_cosine_distance_nonfinite(self, nonfinite):
        step = DeitaFiltering(data_budget=1, pipeline=Pipeline(name="cosine-nonfinite"))
        embeddings = np.ones((513, 4))
        embeddings[512, 0] = nonfinite
        with np.errstate(invalid="ignore"):
            expected = self._full_cosine_distance(embeddings)
            actual = step._cosine_distance(embeddings)
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("budget", [1, 20, 600])
    def test_cosine_process_matches_full_matrix(self, normalize, budget):
        rng = np.random.default_rng(71)
        embeddings = rng.normal(size=(513, 64))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings *= rng.uniform(0.85, 1.2, size=(513, 1))
        inputs = [
            {
                "id": i,
                "embedding": embedding.tolist(),
                "evol_instruction_score": float(rng.uniform(1, 5)),
                "evol_response_score": float(rng.uniform(1, 5)),
            }
            for i, embedding in enumerate(embeddings)
        ]
        step = DeitaFiltering(
            data_budget=budget,
            diversity_threshold=0.65,
            normalize_embeddings=normalize,
            pipeline=Pipeline(name="cosine-process"),
        )
        step.load()
        reference = step._compute_deita_score(copy.deepcopy(inputs))
        vectors = np.array([row["embedding"] for row in reference])
        if normalize:
            vectors = step._normalize_embeddings(vectors)
        distances = self._full_cosine_distance(vectors)
        for row, distance in zip(reference, distances):
            row["nearest_neighbor_distance"] = distance
        reference.sort(key=lambda row: row["deita_score"], reverse=True)
        expected = [
            row for row in reference if row["nearest_neighbor_distance"] >= 0.65
        ][:budget]
        actual = next(step.process(copy.deepcopy(inputs)))
        assert [row["id"] for row in actual] == [row["id"] for row in expected]
        for actual_row, expected_row in zip(actual, expected):
            assert {
                key: value
                for key, value in actual_row.items()
                if key != "nearest_neighbor_distance"
            } == {
                key: value
                for key, value in expected_row.items()
                if key != "nearest_neighbor_distance"
            }
        np.testing.assert_allclose(
            [row["nearest_neighbor_distance"] for row in actual],
            [row["nearest_neighbor_distance"] for row in expected],
            rtol=0,
            atol=1e-12,
        )

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
