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

from distilabel.llms.base import LLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.evol_instruct.generator import (
    EvolInstructGenerator,
)
from distilabel.steps.tasks.evol_instruct.utils import (
    GENERATION_MUTATION_TEMPLATES,
)


class TestEvolInstructGenerator:
    def test_passing_pipeline(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstructGenerator(
            name="task", llm=dummy_llm, num_instructions=2, pipeline=pipeline
        )
        assert task.name == "task"
        assert task.llm is dummy_llm
        assert task.num_instructions == 2
        assert task.mutation_templates == GENERATION_MUTATION_TEMPLATES
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self, dummy_llm: LLM) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            task = EvolInstructGenerator(
                name="task", llm=dummy_llm, num_instructions=2, pipeline=pipeline
            )
            assert task.name == "task"
            assert task.llm is dummy_llm
        assert task.pipeline == pipeline

    def test_with_errors(
        self, caplog: pytest.LogCaptureFixture, dummy_llm: LLM
    ) -> None:
        with pytest.raises(
            ValidationError, match="num_instructions\n  Field required \\[type=missing"
        ):
            EvolInstructGenerator(
                name="task", pipeline=Pipeline(name="unit-test-pipeline")
            )  # type: ignore

        EvolInstructGenerator(name="task", llm=dummy_llm, num_instructions=2)
        assert "Step 'task' hasn't received a pipeline" in caplog.text

    def test_process(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstructGenerator(
            name="task",
            llm=dummy_llm,
            num_instructions=1,
            min_length=1,
            max_length=10,
            pipeline=pipeline,
        )
        task.load()
        assert list(task.process()) == [
            (
                [
                    {
                        "instruction": "output",
                        "model_name": "test",
                    }
                ],
                True,
            )
        ]

    def test_process_generate_answers(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstructGenerator(
            name="task",
            llm=dummy_llm,
            num_instructions=1,
            min_length=1,
            max_length=10,
            generate_answers=True,
            pipeline=pipeline,
        )
        task.load()
        assert list(task.process()) == [
            (
                [
                    {
                        "instruction": "output",
                        "answer": "output",
                        "model_name": "test",
                    }
                ],
                True,
            )
        ]

    def test_serialization(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolInstructGenerator(
            name="task", llm=dummy_llm, num_instructions=2, pipeline=pipeline
        )
        task.load()

        assert task.dump() == {
            "name": "task",
            "llm": {
                "generation_kwargs": {},
                "type_info": {
                    "module": task.llm.__class__.__module__,
                    "name": task.llm.__class__.__name__,
                },
            },
            "add_raw_output": True,
            "input_mappings": task.input_mappings,
            "output_mappings": task.output_mappings,
            "resources": {
                "cpus": None,
                "gpus": None,
                "memory": None,
                "replicas": 1,
                "resources": None,
            },
            "batch_size": task.batch_size,
            "num_instructions": task.num_instructions,
            "generate_answers": task.generate_answers,
            "mutation_templates": {
                "FRESH_START": "Write one question or request containing one or more of the following words: <PROMPT>",
                "CONSTRAINTS": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nPlease add one more constraints/requirements into '#The Given Prompt#'\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "DEEPENING": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nIf #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "CONCRETIZING": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nPlease replace general concepts with more specific concepts.\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "INCREASED_REASONING_STEPS": "I want you act as a Prompt Rewriter.\n\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\n\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n\nYou SHOULD complicate the given prompt using the following method: \nIf #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.\n\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n\n#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n\n",
                "BREADTH": "I want you act as a Prompt Creator.\n\nYour goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\n\nThis new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\n\nThe LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\n\nThe #Created Prompt# must be reasonable and must be understood and responded by humans.\n\n'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\n\n#Given Prompt#:\n<PROMPT>\n#Created Prompt#:\n\n",
            },
            "num_generations": task.num_generations,
            "group_generations": task.group_generations,
            "min_length": task.min_length,
            "max_length": task.max_length,
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
                    "description": "The number of rows that will contain the batches generated by the step.",
                    "name": "batch_size",
                    "optional": True,
                },
                {
                    "name": "llm",
                    "runtime_parameters_info": [
                        {
                            "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`.",
                            "keys": [],
                            "name": "generation_kwargs",
                        },
                    ],
                },
                {
                    "description": "Whether to include the raw output of the LLM in the key `raw_output_<TASK_NAME>` of the `distilabel_metadata` dictionary output column",
                    "name": "add_raw_output",
                    "optional": True,
                },
                {
                    "name": "num_generations",
                    "optional": True,
                    "description": "The number of generations to be produced per input.",
                },
                {
                    "name": "min_length",
                    "optional": True,
                    "description": "Defines the length (in bytes) that the generated instruction needs to be higher than, to be considered valid.",
                },
                {
                    "name": "max_length",
                    "optional": True,
                    "description": "Defines the length (in bytes) that the generated instruction needs to be lower than, to be considered valid.",
                },
                {
                    "name": "seed",
                    "optional": True,
                    "description": "As `numpy` is being used in order to randomly pick a mutation method, then is nice to seed a random seed.",
                },
            ],
            "type_info": {
                "module": EvolInstructGenerator.__module__,
                "name": EvolInstructGenerator.__name__,
            },
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_task = EvolInstructGenerator.from_dict(task.dump())
            assert isinstance(new_task, EvolInstructGenerator)
