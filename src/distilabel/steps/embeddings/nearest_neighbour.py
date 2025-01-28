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

import importlib.util
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset
from pydantic import Field

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class FaissNearestNeighbour(GlobalStep):
    """Create a `faiss` index to get the nearest neighbours.

    `FaissNearestNeighbour` is a `GlobalStep` that creates a `faiss` index using the Hugging
    Face `datasets` library integration, and then gets the nearest neighbours and the scores
    or distance of the nearest neighbours for each input row.

    Attributes:
        device: the CUDA device ID or a list of IDs to be used. If negative integer, it
            will use all the available GPUs. Defaults to `None`.
        string_factory: the name of the factory to be used to build the `faiss` index.
            Available string factories can be checked here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes.
            Defaults to `None`.
        metric_type: the metric to be used to measure the distance between the points. It's
            an integer and the recommend way to pass it is importing `faiss` and then passing
            one of `faiss.METRIC_x` variables. Defaults to `None`.
        k: the number of nearest neighbours to search for each input row. Defaults to `1`.
        search_batch_size: the number of rows to include in a search batch. The value can
            be adjusted to maximize the resources usage or to avoid OOM issues. Defaults
            to `50`.
        train_size: If the index needs a training step, specifies how many vectors will be
            used to train the index.

    Runtime parameters:
        - `device`: the CUDA device ID or a list of IDs to be used. If negative integer,
            it will use all the available GPUs. Defaults to `None`.
        - `string_factory`: the name of the factory to be used to build the `faiss` index.
            Available string factories can be checked here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes.
            Defaults to `None`.
        - `metric_type`: the metric to be used to measure the distance between the points.
            It's an integer and the recommend way to pass it is importing `faiss` and then
            passing one of `faiss.METRIC_x` variables. Defaults to `None`.
        - `k`: the number of nearest neighbours to search for each input row. Defaults to `1`.
        - `search_batch_size`: the number of rows to include in a search batch. The value
            can be adjusted to maximize the resources usage or to avoid OOM issues. Defaults
            to `50`.
        - `train_size`: If the index needs a training step, specifies how many vectors will
            be used to train the index.

    Input columns:
        - embedding (`List[Union[float, int]]`): a sentence embedding.

    Output columns:
        - nn_indices (`List[int]`): a list containing the indices of the `k` nearest neighbours
            in the inputs for the row.
        - nn_scores (`List[float]`): a list containing the score or distance to each `k`
            nearest neighbour in the inputs.

    Categories:
        - embedding

    References:
        - [`The Faiss library`](https://arxiv.org/abs/2401.08281)

    Examples:
        Generating embeddings and getting the nearest neighbours:

        ```python
        from distilabel.models import SentenceTransformerEmbeddings
        from distilabel.pipeline import Pipeline
        from distilabel.steps import EmbeddingGeneration, FaissNearestNeighbour, LoadDataFromHub

        with Pipeline(name="hello") as pipeline:
            load_data = LoadDataFromHub(output_mappings={"prompt": "text"})

            embeddings = EmbeddingGeneration(
                embeddings=SentenceTransformerEmbeddings(
                    model="mixedbread-ai/mxbai-embed-large-v1"
                )
            )

            nearest_neighbours = FaissNearestNeighbour()

            load_data >> embeddings >> nearest_neighbours

        if __name__ == "__main__":
            distiset = pipeline.run(
                parameters={
                    load_data.name: {
                        "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                        "split": "test",
                    },
                },
                use_cache=False,
            )
        ```

    Citations:
        ```
        @misc{douze2024faisslibrary,
            title={The Faiss library},
            author={Matthijs Douze and Alexandr Guzhva and Chengqi Deng and Jeff Johnson and Gergely Szilvasy and Pierre-Emmanuel Mazaré and Maria Lomeli and Lucas Hosseini and Hervé Jégou},
            year={2024},
            eprint={2401.08281},
            archivePrefix={arXiv},
            primaryClass={cs.LG},
            url={https://arxiv.org/abs/2401.08281},
        }
        ```
    """

    device: Optional[RuntimeParameter[Union[int, List[int]]]] = Field(
        default=None,
        description="The CUDA device ID or a list of IDs to be used. If negative integer,"
        " it will use all the available GPUs.",
    )
    string_factory: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The name of the factory to be used to build the `faiss` index."
        "Available string factories can be checked here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes.",
    )
    metric_type: Optional[RuntimeParameter[int]] = Field(
        default=None,
        description="The metric to be used to measure the distance between the points. It's"
        " an integer and the recommend way to pass it is importing `faiss` and thenpassing"
        " one of `faiss.METRIC_x` variables.",
    )
    k: Optional[RuntimeParameter[int]] = Field(
        default=1,
        description="The number of nearest neighbours to search for each input row.",
    )
    search_batch_size: Optional[RuntimeParameter[int]] = Field(
        default=50,
        description="The number of rows to include in a search batch. The value can be adjusted"
        " to maximize the resources usage or to avoid OOM issues.",
    )
    train_size: Optional[RuntimeParameter[int]] = Field(
        default=None,
        description="If the index needs a training step, specifies how many vectors will be used to train the index.",
    )

    def load(self) -> None:
        super().load()

        if importlib.util.find_spec("faiss") is None:
            raise ImportError(
                "`faiss` package is not installed. Please install it using `pip install"
                " 'distilabel[faiss-cpu]' or 'distilabel[faiss-gpu]'`."
            )

    @property
    def inputs(self) -> List[str]:
        return ["embedding"]

    @property
    def outputs(self) -> List[str]:
        return ["nn_indices", "nn_scores"]

    def _build_index(self, inputs: List[Dict[str, Any]]) -> Dataset:
        """Builds a `faiss` index using `datasets` integration.

        Args:
            inputs: a list of dictionaries.

        Returns:
            The build `datasets.Dataset` with its `faiss` index.
        """
        dataset = Dataset.from_list(inputs)
        if self.train_size is not None and self.string_factory:
            self._logger.info("🏋️‍♀️ Starting Faiss index training...")
        dataset.add_faiss_index(
            column="embedding",
            device=self.device,  # type: ignore
            string_factory=self.string_factory,
            metric_type=self.metric_type,
            train_size=self.train_size,
        )
        return dataset

    def _save_index(self, dataset: Dataset) -> None:
        """Save the generated Faiss index as an artifact of the step.

        Args:
            dataset: the dataset with the `faiss` index built.
        """
        self.save_artifact(
            name="faiss_index",
            write_function=lambda path: dataset.save_faiss_index(
                index_name="embedding", file=path / "index.faiss"
            ),
            metadata={
                "num_rows": len(dataset),
                "embedding_dim": len(dataset[0]["embedding"]),
            },
        )

    def _search(self, dataset: Dataset) -> Dataset:
        """Search the top `k` nearest neighbours for each row in the dataset.

        Args:
            dataset: the dataset with the `faiss` index built.

        Returns:
            The updated dataset containing the top `k` nearest neighbours for each row,
            as well as the score or distance.
        """

        def add_search_results(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            queries = np.array(examples["embedding"])
            results = dataset.search_batch(
                index_name="embedding",
                queries=queries,
                k=self.k + 1,  # type: ignore
            )
            examples["nn_indices"] = [indices[1:] for indices in results.total_indices]
            examples["nn_scores"] = [scores[1:] for scores in results.total_scores]
            return examples

        return dataset.map(
            add_search_results, batched=True, batch_size=self.search_batch_size
        )

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        dataset = self._build_index(inputs)
        dataset_with_search_results = self._search(dataset)
        self._save_index(dataset)
        yield dataset_with_search_results.to_list()
