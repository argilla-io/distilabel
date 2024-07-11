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

import pytest
from distilabel.llms.openai import OpenAILLM
from distilabel.steps.tasks.magpie.base import MAGPIE_MULTI_TURN_SYSTEM_PROMPT, Magpie

from tests.unit.conftest import DummyMagpieLLM


class TestMagpie:
    def test_raise_value_error_llm_no_magpie_mixin(self) -> None:
        with pytest.raises(
            ValueError,
            match="`Magpie` task can only be used with an `LLM` that uses the `MagpieChatTemplateMixin`",
        ):
            Magpie(llm=OpenAILLM(model="gpt-4", api_key="fake"))  # type: ignore

    def test_outputs(self) -> None:
        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"))

        assert task.outputs == ["conversation"]

        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            only_instruction=True,
        )

        assert task.outputs == ["instruction"]

    def test_process(self) -> None:
        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=1)

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
        ]

    def test_process_with_n_turns(self) -> None:
        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=2)

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "conversation": [
                    {"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
            {
                "conversation": [
                    {"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
            {
                "conversation": [
                    {"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
        ]

    def test_process_with_system_prompt_per_row(self) -> None:
        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=2)

        task.load()

        assert next(
            task.process(
                inputs=[
                    {"system_prompt": "You're a math expert assistant."},
                    {"system_prompt": "You're a florist expert assistant."},
                    {"system_prompt": "You're a plumber expert assistant."},
                ]
            )
        ) == [
            {
                "conversation": [
                    {"role": "system", "content": "You're a math expert assistant."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
            {
                "conversation": [
                    {"role": "system", "content": "You're a florist expert assistant."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
            {
                "conversation": [
                    {"role": "system", "content": "You're a plumber expert assistant."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
            },
        ]

    def test_process_only_instruction(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            only_instruction=True,
        )

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {"instruction": "Hello Magpie"},
            {"instruction": "Hello Magpie"},
            {"instruction": "Hello Magpie"},
        ]

    def test_serialization(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            only_instruction=True,
        )

        assert task.dump() == {
            "llm": {
                "use_magpie_template": True,
                "magpie_pre_query_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
                "generation_kwargs": {},
                "type_info": {
                    "module": "tests.unit.conftest",
                    "name": "DummyMagpieLLM",
                },
            },
            "n_turns": 1,
            "only_instruction": True,
            "system_prompt": None,
            "name": "magpie_0",
            "resources": {
                "replicas": 1,
                "cpus": None,
                "gpus": None,
                "memory": None,
                "resources": None,
            },
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "group_generations": False,
            "add_raw_output": True,
            "num_generations": 1,
            "runtime_parameters_info": [
                {
                    "name": "llm",
                    "runtime_parameters_info": [
                        {
                            "name": "generation_kwargs",
                            "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`.",
                            "keys": [{"name": "kwargs", "optional": False}],
                        }
                    ],
                },
                {
                    "name": "n_turns",
                    "optional": True,
                    "description": "The number of turns to generate for the conversation.",
                },
                {
                    "name": "only_instruction",
                    "optional": True,
                    "description": "Whether to generate only the instruction. If this argument is `True`, then `n_turns` will be ignored.",
                },
                {
                    "name": "system_prompt",
                    "optional": True,
                    "description": "An optional system prompt that can be used to steer the LLM to generate content of certain topic, guide the style, etc.",
                },
                {
                    "name": "resources",
                    "runtime_parameters_info": [
                        {
                            "name": "replicas",
                            "optional": True,
                            "description": "The number of replicas for the step.",
                        },
                        {
                            "name": "cpus",
                            "optional": True,
                            "description": "The number of CPUs assigned to each step replica.",
                        },
                        {
                            "name": "gpus",
                            "optional": True,
                            "description": "The number of GPUs assigned to each step replica.",
                        },
                        {
                            "name": "memory",
                            "optional": True,
                            "description": "The memory in bytes required for each step replica.",
                        },
                        {
                            "name": "resources",
                            "optional": True,
                            "description": "A dictionary containing names of custom resources and the number of those resources required for each step replica.",
                        },
                    ],
                },
                {
                    "name": "input_batch_size",
                    "optional": True,
                    "description": "The number of rows that will contain the batches processed by the step.",
                },
                {
                    "name": "add_raw_output",
                    "optional": True,
                    "description": "Whether to include the raw output of the LLM in the key `raw_output_<TASK_NAME>` of the `distilabel_metadata` dictionary output column",
                },
                {
                    "name": "num_generations",
                    "optional": True,
                    "description": "The number of generations to be produced per input.",
                },
            ],
            "type_info": {
                "module": "distilabel.steps.tasks.magpie.base",
                "name": "Magpie",
            },
        }
