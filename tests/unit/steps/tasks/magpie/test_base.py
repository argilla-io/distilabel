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

import random
from typing import Any, Dict
from unittest import mock

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

    def test_raise_error_if_system_prompts_weights_do_not_sum_to_one(self) -> None:
        with pytest.raises(
            ValueError,
            match="`*If `system_prompts` attribute is a dictionary containing tuples with"
            " the system prompts and their probability of being chosen",
        ):
            Magpie(
                llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
                system_prompt={
                    "system_prompt_1": ("system_prompt", 0.5),
                    "system_prompt_2": ("system_prompt", 0.4),
                },
            )

    def test_outputs(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            only_instruction=True,
        )

        assert task.outputs == ["instruction", "model_name"]

        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=1)

        assert task.outputs == ["instruction", "response", "model_name"]

        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=2)

        assert task.outputs == ["conversation", "model_name"]

        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            system_prompt={
                "system_prompt_1": ("system_prompt", 0.5),
                "system_prompt_2": ("system_prompt", 0.5),
            },
        )

        assert task.outputs == [
            "instruction",
            "response",
            "system_prompt_key",
            "model_name",
        ]

    def test_process(self) -> None:
        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=1)

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "instruction": "Hello Magpie",
                "response": "Hello Magpie",
                "model_name": "test",
            },
            {
                "instruction": "Hello Magpie",
                "response": "Hello Magpie",
                "model_name": "test",
            },
            {
                "instruction": "Hello Magpie",
                "response": "Hello Magpie",
                "model_name": "test",
            },
        ]

    def test_process_with_system_prompt(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            n_turns=2,
            system_prompt="This is a system prompt.",
            include_system_prompt=True,
        )

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "conversation": [
                    {"role": "system", "content": "This is a system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "system", "content": "This is a system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "system", "content": "This is a system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
        ]

    def test_process_with_several_system_prompts(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            n_turns=2,
            system_prompt=[
                "This is a system prompt.",
                "This is another system prompt.",
            ],
            include_system_prompt=True,
        )

        random.seed(42)

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "conversation": [
                    {"role": "system", "content": "This is another system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "system", "content": "This is a system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "system", "content": "This is a system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
        ]

    def test_process_failing_generation_for_some_rows(self) -> None:
        with mock.patch(
            "tests.unit.conftest.DummyMagpieLLM.generate",
            side_effect=[
                [["Hello Magpie"], [None], ["Hello Magpie"]],
                [["Hello Magpie"], ["Hello Magpie"]],
                [["Hello Magpie"], [None]],
                [["Hello Magpie"]],
            ],
        ):
            task = Magpie(
                llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=2
            )

            task.load()

            assert next(task.process(inputs=[{}, {}, {}])) == [
                {
                    "conversation": [
                        {"role": "user", "content": "Hello Magpie"},
                        {"role": "assistant", "content": "Hello Magpie"},
                        {"role": "user", "content": "Hello Magpie"},
                        {"role": "assistant", "content": "Hello Magpie"},
                    ],
                    "model_name": "test",
                },
                {
                    "conversation": [],
                    "model_name": "test",
                },
                {
                    "conversation": [
                        {"role": "user", "content": "Hello Magpie"},
                        {"role": "assistant", "content": "Hello Magpie"},
                    ],
                    "model_name": "test",
                },
            ]

    def test_process_with_n_turns(self) -> None:
        task = Magpie(llm=DummyMagpieLLM(magpie_pre_query_template="llama3"), n_turns=2)

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
        ]

    def test_process_with_end_with_user(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            n_turns=2,
            end_with_user=True,
        )

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
        ]

    def test_process_with_include_system_prompt(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            n_turns=2,
            include_system_prompt=True,
        )

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
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "conversation": [
                    {"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
        ]

    def test_process_with_system_prompt_per_row(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            n_turns=2,
            include_system_prompt=True,
        )

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
                "system_prompt": "You're a math expert assistant.",
                "conversation": [
                    {"role": "system", "content": "You're a math expert assistant."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "system_prompt": "You're a florist expert assistant.",
                "conversation": [
                    {"role": "system", "content": "You're a florist expert assistant."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
            {
                "system_prompt": "You're a plumber expert assistant.",
                "conversation": [
                    {"role": "system", "content": "You're a plumber expert assistant."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello Magpie"},
                ],
                "model_name": "test",
            },
        ]

    def test_process_with_system_prompt_and_probabilities(self) -> None:
        with mock.patch(
            "random.choices",
            side_effect=[
                ["system_prompt_1"],
                ["system_prompt_2"],
                ["system_prompt_1"],
            ],
        ):
            task = Magpie(
                llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
                system_prompt={
                    "system_prompt_1": ("system_prompt", 0.6),
                    "system_prompt_2": ("system_prompt", 0.4),
                },
            )

            task.load()

            assert next(task.process(inputs=[{}, {}, {}])) == [
                {
                    "instruction": "Hello Magpie",
                    "response": "Hello Magpie",
                    "system_prompt_key": "system_prompt_1",
                    "model_name": "test",
                },
                {
                    "instruction": "Hello Magpie",
                    "response": "Hello Magpie",
                    "system_prompt_key": "system_prompt_2",
                    "model_name": "test",
                },
                {
                    "instruction": "Hello Magpie",
                    "response": "Hello Magpie",
                    "system_prompt_key": "system_prompt_1",
                    "model_name": "test",
                },
            ]

    def test_process_only_instruction(self) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            only_instruction=True,
        )

        task.load()

        assert next(task.process(inputs=[{}, {}, {}])) == [
            {
                "instruction": "Hello Magpie",
                "model_name": "test",
            },
            {
                "instruction": "Hello Magpie",
                "model_name": "test",
            },
            {
                "instruction": "Hello Magpie",
                "model_name": "test",
            },
        ]

    @pytest.mark.parametrize(
        "conversation, include_system_prompt, n_turns, expected",
        [
            (
                [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello user"},
                ],
                False,
                1,
                {"instruction": "Hello Magpie", "response": "Hello user"},
            ),
            (
                [
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello user"},
                ],
                False,
                1,
                {"instruction": "Hello Magpie", "response": "Hello user"},
            ),
            (
                [
                    {"role": "system", "content": "This is a system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello user"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm fine thank you."},
                ],
                True,
                2,
                {
                    "conversation": [
                        {"role": "system", "content": "This is a system prompt."},
                        {"role": "user", "content": "Hello Magpie"},
                        {"role": "assistant", "content": "Hello user"},
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I'm fine thank you."},
                    ],
                },
            ),
            (
                [
                    {"role": "system", "content": "This is a system prompt."},
                    {"role": "user", "content": "Hello Magpie"},
                    {"role": "assistant", "content": "Hello user"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm fine thank you."},
                ],
                False,
                2,
                {
                    "conversation": [
                        {"role": "user", "content": "Hello Magpie"},
                        {"role": "assistant", "content": "Hello user"},
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I'm fine thank you."},
                    ],
                },
            ),
            (
                [],
                False,
                1,
                {"instruction": None, "response": None},
            ),
            (
                [],
                False,
                2,
                {"conversation": []},
            ),
        ],
    )
    def test_prepare_conversation_outputs(
        self,
        conversation,
        include_system_prompt: bool,
        n_turns: int,
        expected: Dict[str, Any],
    ) -> None:
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            n_turns=n_turns,
            include_system_prompt=include_system_prompt,
        )
        assert task._prepare_conversation_outputs([conversation], []) == [expected]

    @pytest.mark.parametrize(
        "system_prompt, n_turns, inputs, random_choices_return, expected_prepared_inputs, expected_system_prompt_keys",
        [
            (
                None,
                1,
                [{"system_prompt": "Custom system prompt."}],
                None,
                [[{"role": "system", "content": "Custom system prompt."}]],
                [],
            ),
            (
                ["Prompt A", "Prompt B"],
                1,
                [{}],
                ["Prompt A"],
                [[{"role": "system", "content": "Prompt A"}]],
                [],
            ),
            (
                {"Key1": "Prompt 1", "Key2": "Prompt 2"},
                1,
                [{}],
                ["Key1"],
                [[{"role": "system", "content": "Prompt 1"}]],
                ["Key1"],
            ),
            (
                {"Key1": ("Prompt 1", 0.7), "Key2": ("Prompt 2", 0.3)},
                1,
                [{}],
                ["Key1"],
                [[{"role": "system", "content": "Prompt 1"}]],
                ["Key1"],
            ),
            (
                None,
                2,
                [{}],
                None,
                [[{"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT}]],
                [],
            ),
            (
                None,
                1,
                [{}],
                None,
                [[]],
                [],
            ),
        ],
    )
    def test_prepare_inputs_for_instruction_generation(
        self,
        system_prompt,
        n_turns,
        inputs,
        random_choices_return,
        expected_prepared_inputs,
        expected_system_prompt_keys,
    ):
        task = Magpie(
            llm=DummyMagpieLLM(magpie_pre_query_template="llama3"),
            n_turns=n_turns,
            system_prompt=system_prompt,
        )

        with mock.patch("random.choices") as mock_choices:
            if random_choices_return is not None:
                mock_choices.return_value = random_choices_return

            prepared_inputs, system_prompt_keys = (
                task._prepare_inputs_for_instruction_generation(inputs)
            )

        assert prepared_inputs == expected_prepared_inputs
        assert system_prompt_keys == expected_system_prompt_keys

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
                "use_offline_batch_generation": False,
                "offline_batch_generation_block_until_done": None,
                "jobs_ids": None,
                "type_info": {
                    "module": "tests.unit.conftest",
                    "name": "DummyMagpieLLM",
                },
            },
            "n_turns": 1,
            "end_with_user": False,
            "include_system_prompt": False,
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
            "add_raw_input": True,
            "num_generations": 1,
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
                            "name": "use_offline_batch_generation",
                            "optional": True,
                            "description": "Whether to use the `offline_batch_generate` method to generate the responses.",
                        },
                        {
                            "name": "offline_batch_generation_block_until_done",
                            "optional": True,
                            "description": "If provided, then polling will be done until the `ofline_batch_generate` method is able to retrieve the results. The value indicate the time to wait between each polling.",
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
                    "description": "An optional system prompt, or a list of system prompts from which a random one will be chosen, or a dictionary of system prompts from which a random one will be choosen, or a dictionary of system prompts with their probability of being chosen. The random system prompt will be chosen per input/output batch. This system prompt can be used to guide the generation of the instruct LLM and steer it to generate instructions of a certain topic.",
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
                    "name": "add_raw_input",
                    "optional": True,
                    "description": "Whether to include the raw input of the LLM in the key `raw_input_<TASK_NAME>` of the `distilabel_metadata` dictionary column",
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
