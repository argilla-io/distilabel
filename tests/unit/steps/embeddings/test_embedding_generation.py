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

from distilabel.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from distilabel.steps.embeddings.embedding_generation import EmbeddingGeneration


class TestEmbeddingGeneration:
    def test_process(self) -> None:
        step = EmbeddingGeneration(
            embeddings=SentenceTransformerEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        step.load()

        results = next(
            step.process(
                inputs=[
                    {"text": "Hello, how are you?"},
                    {"text": "What a nice day!"},
                    {"text": "I hear that llamas are very popular now."},
                ]
            )
        )

        step.unload()

        for result, text in zip(
            results,
            [
                "Hello, how are you?",
                "What a nice day!",
                "I hear that llamas are very popular now.",
            ],
        ):
            assert len(result["embedding"]) == 384
            assert result["text"] == text
            assert result["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
