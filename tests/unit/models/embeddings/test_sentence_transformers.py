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

from distilabel.models.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings,
)


class TestSentenceTransformersEmbeddings:
    def test_model_name(self) -> None:
        embeddings = SentenceTransformerEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        assert embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_encode(self) -> None:
        embeddings = SentenceTransformerEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings.load()

        results = embeddings.encode(
            inputs=[
                "Hello, how are you?",
                "What a nice day!",
                "I hear that llamas are very popular now.",
            ]
        )

        for result in results:
            assert len(result) == 384
