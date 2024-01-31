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

from unittest import mock

import together
from distilabel.llm.together import TogetherInferenceLLM
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask


class TestTogetherInferenceLLM:
    def test_available_models(self) -> None:
        together.Models.list = mock.MagicMock(
            return_value=[
                {"name": "togethercomputer/llama-2-7b", "display_type": "chat"}
            ]
        )
        llm = TogetherInferenceLLM(
            model="togethercomputer/llama-2-7b",
            task=TextGenerationTask(),
            api_key="api.key",
        )
        llm.available_models = ["togethercomputer/llama-2-7b"]

    def test_inference_kwargs(self) -> None:
        together.Models.list = mock.MagicMock(
            return_value=[
                {"name": "togethercomputer/llama-2-7b", "display_type": "chat"}
            ]
        )
        llm = TogetherInferenceLLM(
            model="togethercomputer/llama-2-7b",
            task=TextGenerationTask(),
            api_key="api.key",
            max_new_tokens=256,
            repetition_penalty=1.1,
            temperature=0.8,
            top_p=1.0,
            top_k=50,
            stop=["/n"],
            logprobs=10,
        )

        assert llm.model_name == "togethercomputer/llama-2-7b"
        assert llm.max_new_tokens == 256
        assert llm.repetition_penalty == 1.1
        assert llm.temperature == 0.8
        assert llm.top_p == 1.0
        assert llm.top_k == 50
        assert llm.stop == ["/n"]
        assert llm.logprobs == 10

    def test__generate_single_output(self) -> None:
        together.Models.list = mock.MagicMock(
            return_value=[
                {"name": "togethercomputer/llama-2-7b", "display_type": "chat"}
            ]
        )
        llm = TogetherInferenceLLM(
            model="togethercomputer/llama-2-7b",
            task=TextGenerationTask(),
            api_key="api.key",
        )

        prompt = Prompt(
            system_prompt="You are a helpful assistant", formatted_prompt="What's 2+2?"
        ).format_as("llama2")
        assert isinstance(prompt, str)

        together.Complete.create = mock.MagicMock(
            return_value={"output": {"choices": [{"text": "4"}]}}
        )

        llm_output = llm._generate_single_output(prompt)
        assert llm_output == {
            "model_name": llm.model_name,
            "prompt_used": prompt,
            "raw_output": "4",
            "parsed_output": {"generations": "4"},
        }

    def test__generate(self) -> None:
        together.Models.list = mock.MagicMock(
            return_value=[
                {"name": "togethercomputer/llama-2-7b", "display_type": "chat"}
            ]
        )
        llm = TogetherInferenceLLM(
            model="togethercomputer/llama-2-7b",
            task=TextGenerationTask(system_prompt="You are a helpful assistant"),
            api_key="api.key",
            prompt_format="llama2",
        )

        together.Complete.create = mock.MagicMock(
            return_value={"output": {"choices": [{"text": "4"}]}}
        )

        prompt = "What's 2+2?"
        llm_output = llm._generate(
            [{"input": prompt}, {"input": prompt}], num_generations=2
        )
        assert isinstance(llm_output, list)
        assert len(llm_output) == 2
        assert llm_output[0][0] == {
            "model_name": llm.model_name,
            "prompt_used": Prompt(
                system_prompt="You are a helpful assistant",
                formatted_prompt="What's 2+2?",
            ).format_as("llama2"),
            "raw_output": "4",
            "parsed_output": {"generations": "4"},
        }
