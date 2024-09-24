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

from distilabel.embeddings.llamacpp import LlamaCppEmbeddings


class TestLlamaCppEmbeddings:
    model_name = "all-MiniLM-L6-v2-Q2_K.gguf"
    repo_id = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"

    def test_model_name(self) -> None:
        """
        Test if the model name is correctly set.
        """
        embeddings = LlamaCppEmbeddings(model=self.model_name)
        assert embeddings.model_name == self.model_name

    def test_encode(self, local_llamacpp_model_path) -> None:
        """
        Test if the model can generate embeddings.

        Args:
            local_llamacpp_model_path (str): Fixture providing the local model path.
        """
        embeddings = LlamaCppEmbeddings(model=local_llamacpp_model_path)
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]
        embeddings.load()
        results = embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result["embedding"]) == 384

    def test_load_model_from_local(self, local_llamacpp_model_path):
        """
        Test if the model can be loaded from a local file and generate embeddings.

        Args:
            local_llamacpp_model_path (str): Fixture providing the local model path.
        """
        embeddings = LlamaCppEmbeddings(model=local_llamacpp_model_path)
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]
        embeddings.load()
        # Test if the model is loaded by generating an embedding
        results = embeddings.encode(inputs=inputs)

        embeddings.load()
        results = embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result["embedding"]) == 384

    def test_load_model_from_repo(self):
        """
        Test if the model can be loaded from a Hugging Face repository.
        """
        embeddings = LlamaCppEmbeddings(
            hub_repository_id=self.repo_id, model=self.model_name
        )
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]

        embeddings.load()
        # Test if the model is loaded by generating an embedding
        results = embeddings.encode(inputs=inputs)

        embeddings.load()
        results = embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result["embedding"]) == 384
