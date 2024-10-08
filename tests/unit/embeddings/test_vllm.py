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

from unittest.mock import MagicMock, Mock

from distilabel.embeddings.vllm import vLLMEmbeddings


# @patch("vllm.entrypoints.LLM")
class TestSentenceTransformersEmbeddings:
    model_name = "group/model-name"

    def test_model_name(self) -> None:
        embeddings = vLLMEmbeddings(model=self.model_name)

        assert embeddings.model_name == self.model_name

    def test_encode(self) -> None:
        embeddings = vLLMEmbeddings(model=self.model_name)

        # the loading should be done here, it's just mocked
        # embeddings.load()
        embeddings._model = MagicMock()

        mocked_response = Mock(outputs=Mock(embedding=[0.1] * 10))
        embeddings._model.encode = Mock(
            side_effect=lambda x: [mocked_response for _ in range(len(x))]
        )

        results = embeddings.encode(
            inputs=[
                "Hello, how are you?",
                "What a nice day!",
                "I hear that llamas are very popular now.",
            ]
        )

        for result in results:
            assert len(result) == 10
