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

import pytest

from distilabel.steps.filtering.embedding import EmbeddingDedup

SAMPLE_DATA = [
    {
        "persona": "A chemistry student or academic researcher interested in inorganic or physical chemistry, likely at an advanced undergraduate or graduate level, studying acid-base interactions and chemical bonding.",
        "embedding": [
            0.018477669046149742,
            -0.03748236608841726,
            0.001919870620352492,
            0.024918478063770535,
            0.02348063521315178,
            0.0038251285566308375,
            -0.01723884983037716,
            0.02881971942372201,
        ],
        "nn_indices": [0, 1],
        "nn_scores": [
            0.9164746999740601,
            0.782106876373291,
        ],
    },
    {
        "persona": "A music teacher or instructor focused on theoretical and practical piano lessons.",
        "embedding": [
            -0.0023464179614082125,
            -0.07325472251663565,
            -0.06058678419516501,
            -0.02100326928586996,
            -0.013462744792362657,
            0.027368447064244242,
            -0.003916070100455717,
            0.01243614518480423,
        ],
        "nn_indices": [0, 2],
        "nn_scores": [
            0.7552462220191956,
            0.7261884808540344,
        ],
    },
    {
        "persona": "A classical guitar teacher or instructor, likely with experience teaching beginners, who focuses on breaking down complex music notation into understandable steps for their students.",
        "embedding": [
            -0.01630817942328242,
            -0.023760151552345232,
            -0.014249650090627883,
            -0.005713686451446624,
            -0.016033059279131567,
            0.0071440908501058786,
            -0.05691099643425161,
            0.01597412704817784,
        ],
        "nn_indices": [1, 2],
        "nn_scores": [
            0.8107735514640808,
            0.7172299027442932,
        ],
    },
    {
        "persona": "A classical guitar teacher or instructor, likely with experience teaching beginners, who focuses on breaking down complex music notation into understandable steps for their students.",
        "embedding": [
            -0.01630817942328242,
            -0.023760151552345232,
            -0.014249650090627883,
            -0.005713686451446624,
            -0.016033059279131567,
            0.0071440908501058786,
            -0.05691099643425161,
            0.01597412704817784,
        ],
        "nn_indices": [],
        "nn_scores": [],
    },
]


class TestEmbeddingDedup:
    @pytest.mark.parametrize(
        "threshold, keep_row_after_embedding_filtering",
        [(0.1, 1), (0.9, 3), (0.99999, 4)],
    )
    def test_process(
        self, threshold: float, keep_row_after_embedding_filtering: int
    ) -> None:
        step = EmbeddingDedup(threshold=threshold)
        step.load()
        result = next(step.process(SAMPLE_DATA))
        duplicated = [r["keep_row_after_embedding_filtering"] for r in result]
        assert sum(duplicated) == keep_row_after_embedding_filtering
