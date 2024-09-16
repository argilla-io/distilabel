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
from distilabel.steps.tasks.magpie.generator import MagpieGenerator
from tests.unit.conftest import DummyMagpieLLM


class TestMagpieGenerator:
    def test_raise_value_error_llm_no_magpie_mixin(self) -> None:
        with pytest.raises(
            ValueError,
            match="`Magpie` task can only be used with an `LLM` that uses the `MagpieChatTemplateMixin`",
        ):
            MagpieGenerator(llm=OpenAILLM(model="gpt-4", api_key="fake"))  # type: ignore

    def test_serialization(self) -> None:
        task = MagpieGenerator(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"))

        assert task.dump() == {
            "llm": {
                "use_magpie_template": True,
                "magpie_pre_query_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
                "generation_kwargs": {},
                "jobs_ids": None,
                "offline_batch_generation_block_until_done": None,
                "use_offline_batch_generation": False,
                "type_info": {
                    "module": "tests.unit.conftest",
                    "name": "DummyMagpieLLM",
                },
            },
            "n_turns": 1,
            "end_with_user": False,
            "include_system_prompt": False,
            "only_instruction": False,
            "system_prompt": None,
            "name": "magpie_generator_0",
            "resources": {
                "replicas": 1,
                "cpus": None,
                "gpus": None,
                "memory": None,
                "resources": None,
            },
            "input_mappings": {},
            "output_mappings": {},
            "batch_size": 50,
            "group_generations": False,
            "add_raw_output": True,
            "add_raw_input": True,
            "num_generations": 1,
            "num_rows": None,
            "use_default_structured_output": False,
            "runtime_parameters_info": [
                {
                    "name": "llm",
                    "runtime_parameters_info": [
                        {
                            "name": "generation_kwargs",
                            "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`.",
                            "keys": [{"name": "kwargs", "optional": False}],
                        },
                        {
                            "description": "Whether to use the `offline_batch_generate` method to "
                            "generate the responses.",
                            "name": "use_offline_batch_generation",
                            "optional": True,
                        },
                        {
                            "description": "If provided, then polling will be done until the "
                            "`ofline_batch_generate` method is able to retrieve the "
                            "results. The value indicate the time to wait between each "
                            "polling.",
                            "name": "offline_batch_generation_block_until_done",
                            "optional": True,
                        },
                    ],
                },
                {
                    "name": "n_turns",
                    "optional": True,
                    "description": "The number of turns to generate for the conversation.",
                },
                {
                    "name": "end_with_user",
                    "optional": True,
                    "description": "Whether the conversation should end with a user message.",
                },
                {
                    "name": "include_system_prompt",
                    "optional": True,
                    "description": "Whether to include the system prompt used in the generated conversation.",
                },
                {
                    "name": "only_instruction",
                    "optional": True,
                    "description": "Whether to generate only the instruction. If this argument is `True`, then `n_turns` will be ignored.",
                },
                {
                    "name": "system_prompt",
                    "optional": True,
                    "description": "An optional system prompt or list of system prompts that can be used to steer the LLM to generate content of certain topic, guide the style, etc.",
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
                    "name": "batch_size",
                    "optional": True,
                    "description": "The number of rows that will contain the batches generated by the step.",
                },
                {
                    "name": "add_raw_output",
                    "optional": True,
                    "description": "Whether to include the raw output of the LLM in the key `raw_output_<TASK_NAME>` of the `distilabel_metadata` dictionary output column",
                },
                {
                    "description": "Whether to include the raw input of the LLM in the key `raw_input_<TASK_NAME>` of the `distilabel_metadata` dictionary column",
                    "name": "add_raw_input",
                    "optional": True,
                },
                {
                    "name": "num_generations",
                    "optional": True,
                    "description": "The number of generations to be produced per input.",
                },
                {
                    "name": "num_rows",
                    "optional": False,
                    "description": "The number of rows to generate.",
                },
            ],
            "type_info": {
                "module": "distilabel.steps.tasks.magpie.generator",
                "name": "MagpieGenerator",
            },
        }
