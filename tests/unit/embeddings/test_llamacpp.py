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

from distilabel.embeddings import LlamaCppEmbeddings

"""
To test with CPU only, run the following command:
pytest tests/unit/embeddings/test_llamacpp.py --cpu-only

"""


class TestLlamaCppEmbeddings:
    model_path = "Downloads/all-MiniLM-L6-v2-Q2_K.gguf"
    repo_id = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
    hub_model = "all-MiniLM-L6-v2-Q5_K_M.gguf"

    @pytest.fixture(autouse=True)
    def setup_embeddings(self, local_llamacpp_model_path, use_cpu):
        """
        Fixture to set up embeddings for each test, considering CPU usage.
        """
        n_gpu_layers = 0 if use_cpu else -1
        self.embeddings = LlamaCppEmbeddings(
            model_path=local_llamacpp_model_path, n_gpu_layers=n_gpu_layers
        )
        self.embeddings.load()

    def test_model_name(self, local_llamacpp_model_path) -> None:
        """
        Test if the model name is correctly set.
        """
        assert self.embeddings.model_name == local_llamacpp_model_path

    def test_encode(self) -> None:
        """
        Test if the model can generate embeddings.
        """
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]
        results = self.embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result) == 384

    def test_load_model_from_local(self):
        """
        Test if the model can be loaded from a local file and generate embeddings.

        Args:
            local_llamacpp_model_path (str): Fixture providing the local model path.
        """

        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]

        # Test if the model is loaded by generating an embedding
        results = self.embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result) == 384

    def test_load_model_from_repo(self, use_cpu):
        """
        Test if the model can be loaded from a Hugging Face repository.
        """
        n_gpu_layers = 0 if use_cpu else -1
        embeddings = LlamaCppEmbeddings(
            repo_id=self.repo_id,
            normalize_embeddings=True,
            model_path=self.hub_model,
            n_gpu_layers=n_gpu_layers,
        )
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]

        embeddings.load()
        # Test if the model is loaded by generating an embedding
        results = embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result) == 384

    def test_normalize_embeddings(self, use_cpu):
        """
        Test if embeddings are normalized when normalize_embeddings is True.
        """
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]
        n_gpu_layers = 0 if use_cpu else -1
        embeddings = LlamaCppEmbeddings(
            repo_id=self.repo_id,
            normalize_embeddings=True,
            model_path=self.hub_model,
            n_gpu_layers=n_gpu_layers,
        )
        embeddings.load()
        results = embeddings.encode(inputs=inputs)

        for result in results:
            # Check if the embedding is normalized (L2 norm should be close to 1)
            norm = np.linalg.norm(result)
            assert np.isclose(
                norm, 1.0, atol=1e-6
            ), f"Norm is {norm}, expected close to 1.0"

    def test_normalize_embeddings_false(self):
        """
        Test if embeddings are not normalized when normalize_embeddings is False.
        """

        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]

        results = self.embeddings.encode(inputs=inputs)

        for result in results:
            # Check if the embedding is not normalized (L2 norm should not be close to 1)
            norm = np.linalg.norm(result)
            assert not np.isclose(
                norm, 1.0, atol=1e-6
            ), f"Norm is {norm}, expected not close to 1.0"

        # Additional check: ensure that at least one embedding has a norm significantly different from 1
        norms = [np.linalg.norm(result) for result in results]
        assert any(
            not np.isclose(norm, 1.0, atol=0.1) for norm in norms
        ), "Expected at least one embedding with norm not close to 1.0"

    def test_encode_batch(self) -> None:
        """
        Test if the model can generate embeddings for batches of inputs.
        """
        # Test with different batch sizes
        batch_sizes = [1, 2, 5, 10]
        for batch_size in batch_sizes:
            inputs = [f"This is test sentence {i}" for i in range(batch_size)]
            results = self.embeddings.encode(inputs=inputs)

            assert (
                len(results) == batch_size
            ), f"Expected {batch_size} results, got {len(results)}"
            for result in results:
                assert (
                    len(result) == 384
                ), f"Expected embedding dimension 384, got {len(result)}"

        # Test with a large batch to ensure it doesn't cause issues
        large_batch = ["Large batch test" for _ in range(100)]
        large_results = self.embeddings.encode(inputs=large_batch)
        assert (
            len(large_results) == 100
        ), f"Expected 100 results for large batch, got {len(large_results)}"

    def test_encode_batch_consistency(self) -> None:
        """
        Test if the model produces consistent embeddings for the same input in different batch sizes.

        Args:
            local_llamacpp_model_path (str): Fixture providing the local model path.
        """
        input_text = "This is a test sentence for consistency"

        # Generate embedding individually
        single_result = self.embeddings.encode([input_text])[0]

        # Generate embedding as part of a batch
        batch_result = self.embeddings.encode([input_text, "Another sentence"])[0]

        # Compare the embeddings
        assert np.allclose(
            single_result, batch_result, atol=1e-5
        ), "Embeddings are not consistent between single and batch processing"
