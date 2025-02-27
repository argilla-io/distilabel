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

from typing import TYPE_CHECKING

import faiss
import numpy as np

from distilabel.pipeline import Pipeline
from distilabel.steps import FaissNearestNeighbour, LoadDataFromDicts, StepInput, step
from distilabel.steps.filtering.embedding import EmbeddingDedup

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


SAMPLE_DATA = [
    {
        "text": "A chemistry student or academic researcher interested in inorganic or physical chemistry, likely at an advanced undergraduate or graduate level, studying acid-base interactions and chemical bonding.",
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
    },
    {
        "text": "A music teacher or instructor focused on theoretical and practical piano lessons.",
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
    },
    {
        "text": "A classical guitar teacher or instructor, likely with experience teaching beginners, who focuses on breaking down complex music notation into understandable steps for their students.",
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
    },
    {
        "text": "A classical guitar teacher or instructor, likely with experience teaching beginners, who focuses on breaking down complex music notation into understandable steps for their students.",
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
    },
]


@step(inputs=["embedding"], outputs=["embedding"])
def NormalizeEmbeddings(inputs: StepInput) -> "StepOutput":
    # Normalize a vector to have length 1
    for input in inputs:
        norm = np.linalg.norm(input["embedding"])
        if norm == 0:
            print("Cannot normalize a zero vector")
            continue
        input["embedding"] = input["embedding"] / norm
    yield inputs


def test_embedding_deduplication() -> None:
    with Pipeline() as pipeline:
        loader = LoadDataFromDicts(
            data=SAMPLE_DATA * 20,
            batch_size=50,
        )
        batch_size = 50

        # NOTE: Guide to choose an index: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        nn = FaissNearestNeighbour(
            k=5,
            metric_type=faiss.METRIC_INNER_PRODUCT,
            search_batch_size=50,
            # string_factory="IVF300_HNSW32,Flat",
            # train_size=len(dataset),
            input_batch_size=batch_size,
        )

        embedding_dedup = EmbeddingDedup(
            threshold=0.99,
            input_batch_size=batch_size,
        )
        normalize = NormalizeEmbeddings()
        loader >> normalize >> nn >> embedding_dedup

    distiset = pipeline.run(use_cache=False)

    ds = distiset["default"]["train"]
    ds_dedup = ds.filter(lambda x: x["keep_row_after_embedding_filtering"])

    assert len(ds_dedup) == 63


if __name__ == "__main__":
    test_embedding_deduplication()
