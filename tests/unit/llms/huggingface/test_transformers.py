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

from typing import Generator

import pytest

from distilabel.llms.huggingface.transformers import TransformersLLM


# load the model just once for all the tests in the module
@pytest.fixture(scope="module")
def transformers_llm() -> Generator[TransformersLLM, None, None]:
    llm = TransformersLLM(
        model="distilabel-internal-testing/tiny-random-mistral",
        model_kwargs={"is_decoder": True},
        cuda_devices=[],
        torch_dtype="float16",
    )
    llm.load()

    yield llm


class TestTransformersLLM:
    def test_model_name(self, transformers_llm: TransformersLLM) -> None:
        assert (
            transformers_llm.model_name
            == "distilabel-internal-testing/tiny-random-mistral"
        )

    def test_generate(self, transformers_llm: TransformersLLM) -> None:
        responses = transformers_llm.generate(
            inputs=[
                [{"role": "user", "content": "Hello, how are you?"}],
                [
                    {
                        "role": "user",
                        "content": "You're GPT2, you're old now but you still serve a purpose which is being used in unit tests.",
                    }
                ],
            ],
            num_generations=3,
        )
        assert len(responses) == 2
        generations = responses[0]["generations"]
        statistics = responses[0]["statistics"]
        assert len(generations) == 3
        assert "input_tokens" in statistics
        assert "output_tokens" in statistics

    def test_get_last_hidden_states(self, transformers_llm: TransformersLLM) -> None:
        inputs = [
            [{"role": "user", "content": "Hello, how are you?"}],
            [{"role": "user", "content": "Hello, you're in a unit test"}],
        ]
        last_hidden_states = transformers_llm.get_last_hidden_states(inputs)  # type: ignore

        assert last_hidden_states[0].shape == (7, 128)
        assert last_hidden_states[1].shape == (10, 128)
