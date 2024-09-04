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

from typing import TYPE_CHECKING, List

import numpy as np
from rich.progress import track
from typing_extensions import override

from distilabel.steps.base import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class EmbeddingDedup(GlobalStep):
    """Deduplicates text using embeddings.

    `EmbeddingDedup` is a Step that detects near-duplicates in datasets, using
    embeddings to compare the similarity between the texts. The typical workflow with this step
    would include having a dataset with embeddings precomputed, and then (possibly using the
    `FaissNearestNeighbour`) using the `nn_indices` and `nn_scores`, determine the texts that
    are duplicate.

    Attributes:
        threshold: the threshold to consider 2 examples as duplicates.
            It's dependent on the type of index that was used to generate the embeddings.
            For example, if the embeddings were generated using cosine similarity, a threshold
            of `0.9` would make all the texts with a cosine similarity above the value
            duplicates. Higher values detect less duplicates in such an index, but that should
            be taken into account when building it. Defaults to `0.9`.

    Input columns:
        - nn_indices (`List[int]`): a list containing the indices of the `k` nearest neighbours
            in the inputs for the row.
        - nn_scores (`List[float]`): a list containing the score or distance to each `k`
            nearest neighbour in the inputs.

    Output columns:
        - keep_row_after_embedding_filtering (`bool`): boolean indicating if the piece `text` is
            not a duplicate i.e. this text should be kept.

    Categories:
        - filtering

    Examples:

        Deduplicate a list of texts using embedding information:

        ```python
        from distilabel.pipeline import Pipeline
        from distilabel.steps import EmbeddingDedup
        from distilabel.steps import LoadDataFromDicts

        with Pipeline() as pipeline:
            data = LoadDataFromDicts(
                data=[
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
                ],
                batch_size=batch_size,
            )
            # In general you should do something like this before the deduplication step, to obtain the
            # `nn_indices` and `nn_scores`. In this case the embeddings are already normalized, so there's
            # no need for it.
            # nn = FaissNearestNeighbour(
            #     k=30,
            #     metric_type=faiss.METRIC_INNER_PRODUCT,
            #     search_batch_size=50,
            #     train_size=len(dataset),              # The number of embeddings to use for training
            #     string_factory="IVF300_HNSW32,Flat"   # To use an index (optional, maybe required for big datasets)
            # )
            # Read more about the `string_factory` here:
            # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

            embedding_dedup = EmbeddingDedup(
                threshold=0.8,
                input_batch_size=batch_size,
            )

            data >> embedding_dedup

        if __name__ == "__main__":
            distiset = pipeline.run(use_cache=False)
            ds = distiset["default"]["train"]
            # Filter out the duplicates
            ds_dedup = ds.filter(lambda x: x["keep_row_after_embedding_filtering"])
        ```
    """

    threshold: float = 0.9

    @property
    def inputs(self) -> List[str]:
        return ["nn_scores", "nn_indices"]

    @property
    def outputs(self) -> List[str]:
        return ["keep_row_after_embedding_filtering"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        rows_to_remove = set()

        for input in track(inputs, description="Running Embedding deduplication..."):
            input["keep_row_after_embedding_filtering"] = True
            indices_scores = np.array(input["nn_scores"]) > self.threshold
            indices = np.array(input["nn_indices"])[indices_scores]
            if len(indices) > 0:  # If there are any rows found over the threshold
                rows_to_remove.update(list(indices))

        # Remove duplicates and get the list of rows to remove
        for idx in rows_to_remove:
            inputs[idx]["keep_row_after_embedding_filtering"] = False

        yield inputs
