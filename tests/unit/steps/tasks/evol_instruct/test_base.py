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
from pydantic import ValidationError

from distilabel.models.llms.base import LLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.evol_instruct.base import (
    EvolInstruct,
)
from distilabel.steps.tasks.evol_instruct.utils import (
    MUTATION_TEMPLATES,
)


class TestEvolInstruct:
    def test_passing_pipeline(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstruct(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        assert task.name == "task"
        assert task.llm is dummy_llm
        assert task.num_evolutions == 2
        assert task.mutation_templates == MUTATION_TEMPLATES
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self, dummy_llm: LLM) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            task = EvolInstruct(
                name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
            )
            assert task.name == "task"
            assert task.llm is dummy_llm
        assert task.pipeline == pipeline

    def test_with_errors(
        self, caplog: pytest.LogCaptureFixture, dummy_llm: LLM
    ) -> None:
        with pytest.raises(
            ValidationError, match="num_evolutions\n  Field required \\[type=missing"
        ):
            EvolInstruct(name="task", pipeline=Pipeline(name="unit-test-pipeline"))  # type: ignore

        EvolInstruct(name="task", llm=dummy_llm, num_evolutions=2)
        assert "Step 'task' hasn't received a pipeline" in caplog.text

    def test_process(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstruct(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        task.load()
        assert list(task.process([{"instruction": "test"}])) == [
            [
                {
                    "instruction": "test",
                    "evolved_instruction": "output",
                    "model_name": "test",
                    "distilabel_metadata": {
                        "statistics_instruction_task": {
                            "input_tokens": [12, 12],
                            "output_tokens": [12, 12],
                        }
                    },
                }
            ]
        ]

    def test_process_store_evolutions(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstruct(
            name="task",
            llm=dummy_llm,
            num_evolutions=2,
            store_evolutions=True,
            pipeline=pipeline,
        )
        task.load()
        assert list(task.process([{"instruction": "test"}])) == [
            [
                {
                    "instruction": "test",
                    "evolved_instructions": ["output", "output"],
                    "model_name": "test",
                    "distilabel_metadata": {
                        "statistics_instruction_task": {
                            "input_tokens": [12, 12],
                            "output_tokens": [12, 12],
                        }
                    },
                }
            ]
        ]

    def test_process_generate_answers(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstruct(
            name="task",
            llm=dummy_llm,
            num_evolutions=2,
            generate_answers=True,
            pipeline=pipeline,
        )
        task.load()
        assert list(task.process([{"instruction": "test"}])) == [
            [
                {
                    "instruction": "test",
                    "evolved_instruction": "output",
                    "answer": "output",
                    "model_name": "test",
                    "distilabel_metadata": {
                        "statistics_answer_task": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        }
                    },
                }
            ]
        ]

    def test_serialization(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstruct(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        task.load()
        assert task.dump() == {
            "name": "task",
            "add_raw_output": True,
            "add_raw_input": True,
            "input_mappings": task.input_mappings,
            "output_mappings": task.output_mappings,
            "inputs": ["instruction"],
            "resources": {
                "cpus": None,
                "gpus": None,
                "memory": None,
                "replicas": 1,
                "resources": None,
            },
            "input_batch_size": task.input_batch_size,
            "llm": {
                "generation_kwargs": {},
                "structured_output": None,
                "jobs_ids": None,
                "offline_batch_generation_block_until_done": None,
                "use_offline_batch_generation": False,
                "n_generations_supported": True,
                "type_info": {
                    "module": task.llm.__module__,
                    "name": task.llm.__class__.__name__,
                },
            },
            "group_generations": task.group_generations,
            "num_generations": task.num_generations,
            "num_evolutions": task.num_evolutions,
            "store_evolutions": task.store_evolutions,
            "generate_answers": task.generate_answers,
            "include_original_instruction": task.include_original_instruction,
            "mutation_templates": {
                "CONSTRAINTS": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nPlease add one more constraints/requirements into '#The Given Prompt#'\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "DEEPENING": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nIf #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "CONCRETIZING": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nPlease replace general concepts with more specific concepts.\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "INCREASED_REASONING_STEPS": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nIf #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "BREADTH": "I want you act as a Prompt Creator.\n\nYour goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\n\nThis new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\n\nThe LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\n\nThe #Created Prompt# must be reasonable and must be understood and responded by humans.\n\n'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\n\n#Given Prompt#:\n<PROMPT>\n#Created Prompt#:\n\n",
            },
            "use_default_structured_output": False,
            "seed": task.seed,
            "runtime_parameters_info": [
                {
                    "name": "resources",
                    "runtime_parameters_info": [
                        {
                            "description": "The number of replicas for the step.",
                            "name": "replicas",
                            "optional": True,
                        },
                        {
                            "description": "The number of CPUs assigned to each step replica.",
                            "name": "cpus",
                            "optional": True,
                        },
                        {
                            "description": "The number of GPUs assigned to each step replica.",
                            "name": "gpus",
                            "optional": True,
                        },
                        {
                            "description": "The memory in bytes required for each step replica.",
                            "name": "memory",
                            "optional": True,
                        },
                        {
                            "description": "A dictionary containing names of custom resources and the number of those resources required for each step replica.",
                            "name": "resources",
                            "optional": True,
                        },
                    ],
                },
                {
                    "description": "The number of rows that will contain the batches processed by the step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
                {
                    "name": "llm",
                    "runtime_parameters_info": [
                        {
                            "name": "generation_kwargs",
                            "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`.",
                            "keys": [],
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
                    "description": "Whether to include the raw output of the LLM in the key `raw_output_<TASK_NAME>` of the `distilabel_metadata` dictionary output column",
                    "name": "add_raw_output",
                    "optional": True,
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
                    "name": "seed",
                    "optional": True,
                    "description": "As `numpy` is being used in order to randomly pick a mutation method, then is nice to seed a random seed.",
                },
            ],
            "use_cache": True,
            "type_info": {
                "module": "distilabel.steps.tasks.evol_instruct.base",
                "name": "EvolInstruct",
            },
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_task = EvolInstruct.from_dict(task.dump())
            assert isinstance(new_task, EvolInstruct)
