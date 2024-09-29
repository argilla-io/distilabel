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

from typing import TYPE_CHECKING, Optional

from distilabel.embeddings.base import Embeddings
from distilabel.knowledge_bases.base import KnowledgeBase
from distilabel.steps.base import Step, StepInput
from distilabel.utils.dicts import group_dicts

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class VectorSearch(Step):
    """Execute Vector Search using a `KnowledgeBase` and a potential `Embeddings`.

    `VectorSearch` is a `Step` that using an `KnowledgeBase` generates sentence
    embeddings for the provided input texts.

    Attributes:
        knowledge_base: the `KnowledgeBase` used to generate the sentence embeddings.

    Input columns:
        - text (Optional[`str`]): The text for which the sentence embedding has to be generated.
        - embedding (`Optional[List[Union[float, int]]]`): The sentence embedding generated for the input text.

    Output columns:
        - dynamic: The dynamic columns of the `KnowledgeBase`.

    Categories:
        - knowledge_base

    Examples:
        Do vector search using a `KnowledgeBase` and a potential `Embeddings`.

        ```python
        from distilabel.embeddings import SentenceTransformerEmbeddings
        from distilabel.knowledge_bases.lancedb import LanceDB
        from distilabel.steps.knowledge_bases.vector_search import VectorSearch

        embedding = SentenceTransformerEmbeddings(
            model="mixedbread-ai/mxbai-embed-large-v1",
        )

        knowledge_base = LanceDB(
            uri="data/sample-lancedb",
            table_name="my_table",
        )

       vector_search = VectorSearch(
           knowledge_base=knowledge_base,
           embeddings=embedding,
           n_retrieved_documents=5
        )

        vector_search.load()
        result = next(vector_search.process([{"text": "Hello, how are you?"}]))
        # [{
        #   'text': 'Hello, how are you?',
        #   'embedding': [0.06209656596183777, -0.015797119587659836, ...],
        #   'knowledge_base_col_1': [10.0],
        #   'knowledge_base_col_2': ['foo']
        # }]
        ```

        Do vector search using a `KnowledgeBase`.

        ```python
        from distilabel.embeddings import SentenceTransformerEmbeddings
        from distilabel.knowledge_bases.lancedb import LanceDB
        from distilabel.steps.knowledge_bases.vector_search import VectorSearch

        knowledge_base = LanceDB(
            uri="data/sample-lancedb",
            table_name="my_table",
        )

        vector_search = VectorSearch(
            knowledge_base=knowledge_base,
            n_retrieved_documents=5
        )

        vector_search.load()
        result = next(embedding_generation.process([{'embedding': [0.06209656596183777, -0.015797119587659836, ...]}]))
        # [{'embedding': [0.06209656596183777, -0.015797119587659836, ...], "knowledge_base_col_1": [10.0], "knowledge_base_col_2": ["foo"]}]
        ```
    """

    knowledge_base: KnowledgeBase
    embeddings: Optional[Embeddings] = None
    n_retrieved_documents: Optional[int] = 5

    @property
    def inputs(self) -> "StepColumns":
        if self.embeddings:
            return ["embedding"]
        return ["text"]

    @property
    def outputs(self) -> "StepColumns":
        return self.knowledge_base.columns

    def load(self) -> None:
        """Loads the `Embeddings` model."""
        super().load()
        self.knowledge_base.load()
        if self.embeddings:
            self.embeddings.load()

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Process the inputs and yield the outputs."""
        if self.embeddings:
            embeddings = self.embeddings.encode(
                inputs=[input["text"] for input in inputs]
            )
            for input, embedding in zip(inputs, embeddings):
                input["embedding"] = embedding
                input["model_name"] = self.embeddings.model_name
        for input in inputs:
            retrieved_documents = self.knowledge_base.query(
                input["embedding"], n_retrieved_documents=self.n_retrieved_documents
            )
            grouped_documents = group_dicts(*retrieved_documents)
            input.update(grouped_documents)
        yield inputs

    def unload(self) -> None:
        """Unloads the `Embeddings` and `KnowledgeBase` models."""
        super().unload()
        self.knowledge_base.unload()
        if self.embeddings:
            self.embeddings.unload()
