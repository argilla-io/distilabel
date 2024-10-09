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

import sys
from dataclasses import field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.base import Task
from tests.unit.conftest import (
    DummyAsyncLLM,
    DummyTask,
    DummyTaskOfflineBatchGeneration,
)

if TYPE_CHECKING:
    pass


class DummyRuntimeLLM(DummyAsyncLLM):
    runtime_parameter: RuntimeParameter[int]
    runtime_parameter_optional: Optional[RuntimeParameter[int]] = field(default=None)


class TestTask:
    def test_model_post_init_raise_valuerror_use_offline_batch_generation(self) -> None:
        with pytest.raises(
            ValidationError,
            match="`DummyTask` task cannot be used with offline batch generation",
        ):
            DummyTask(llm=DummyAsyncLLM(use_offline_batch_generation=True))

    def test_is_global_with_offline_batch_generation(self) -> None:
        task = DummyTaskOfflineBatchGeneration(
            llm=DummyAsyncLLM(use_offline_batch_generation=True)
        )
        assert task.is_global is True

    def test_passing_pipeline(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = DummyTask(name="task", llm=llm, pipeline=pipeline)
        assert task.name == "task"
        assert task.llm is llm
        assert task.num_generations == 1
        assert task.group_generations is False
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            llm = DummyAsyncLLM()
            task = DummyTask(name="task", llm=llm, pipeline=pipeline)
            assert task.name == "task"
            assert task.llm is llm
        assert task.pipeline == pipeline

    def test_with_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        DummyTask(name="task", llm=DummyAsyncLLM())
        assert "Step 'task' hasn't received a pipeline" in caplog.text

        with pytest.raises(
            ValidationError, match="llm\n  Field required \\[type=missing"
        ):
            DummyTask(name="task", pipeline=Pipeline(name="unit-test-pipeline"))  # type: ignore

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class Task with abstract methods format_input, format_output"
            if sys.version_info < (3, 12)
            else "Can't instantiate abstract class Task without an implementation for abstract methods 'format_input', 'format_output'",
        ):
            Task(name="task", llm=DummyAsyncLLM())  # type: ignore

    @pytest.mark.parametrize(
        "input, group_generations, expected",
        [
            (
                [
                    {"instruction": "test_0", "additional_info": "additional_info_0"},
                    {"instruction": "test_1", "additional_info": "additional_info_1"},
                    {"instruction": "test_2", "additional_info": "additional_info_2"},
                ],
                False,
                [
                    {
                        "instruction": "test_0",
                        "additional_info": "additional_info_0",
                        "output": "output",
                        "info_from_input": "additional_info_0",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_0", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_0",
                        "additional_info": "additional_info_0",
                        "output": "output",
                        "info_from_input": "additional_info_0",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_0", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_0",
                        "additional_info": "additional_info_0",
                        "output": "output",
                        "info_from_input": "additional_info_0",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_0", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_1",
                        "additional_info": "additional_info_1",
                        "output": "output",
                        "info_from_input": "additional_info_1",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_1", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_1",
                        "additional_info": "additional_info_1",
                        "output": "output",
                        "info_from_input": "additional_info_1",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_1", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_1",
                        "additional_info": "additional_info_1",
                        "output": "output",
                        "info_from_input": "additional_info_1",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_1", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_2",
                        "additional_info": "additional_info_2",
                        "output": "output",
                        "info_from_input": "additional_info_2",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_2", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_2",
                        "additional_info": "additional_info_2",
                        "output": "output",
                        "info_from_input": "additional_info_2",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_2", "role": "user"},
                            ],
                        },
                    },
                    {
                        "instruction": "test_2",
                        "additional_info": "additional_info_2",
                        "output": "output",
                        "info_from_input": "additional_info_2",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "raw_output_task": "output",
                            "raw_input_task": [
                                {"content": "", "role": "system"},
                                {"content": "test_2", "role": "user"},
                            ],
                        },
                    },
                ],
            ),
            (
                [
                    {"instruction": "test_0", "additional_info": "additional_info_0"},
                    {"instruction": "test_1", "additional_info": "additional_info_1"},
                    {"instruction": "test_2", "additional_info": "additional_info_2"},
                ],
                True,
                [
                    {
                        "instruction": "test_0",
                        "additional_info": "additional_info_0",
                        "output": ["output", "output", "output"],
                        "info_from_input": [
                            "additional_info_0",
                            "additional_info_0",
                            "additional_info_0",
                        ],
                        "model_name": "test",
                        "distilabel_metadata": [
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_0",
                                        "role": "user",
                                    },
                                ],
                            },
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_0",
                                        "role": "user",
                                    },
                                ],
                            },
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_0",
                                        "role": "user",
                                    },
                                ],
                            },
                            # {"raw_output_task": "output"},
                            # {"raw_output_task": "output"},
                            # {"raw_output_task": "output"},
                        ],
                    },
                    {
                        "instruction": "test_1",
                        "additional_info": "additional_info_1",
                        "output": ["output", "output", "output"],
                        "info_from_input": [
                            "additional_info_1",
                            "additional_info_1",
                            "additional_info_1",
                        ],
                        "model_name": "test",
                        "distilabel_metadata": [
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_1",
                                        "role": "user",
                                    },
                                ],
                            },
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_1",
                                        "role": "user",
                                    },
                                ],
                            },
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_1",
                                        "role": "user",
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "instruction": "test_2",
                        "additional_info": "additional_info_2",
                        "output": ["output", "output", "output"],
                        "info_from_input": [
                            "additional_info_2",
                            "additional_info_2",
                            "additional_info_2",
                        ],
                        "model_name": "test",
                        "distilabel_metadata": [
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_2",
                                        "role": "user",
                                    },
                                ],
                            },
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_2",
                                        "role": "user",
                                    },
                                ],
                            },
                            {
                                "raw_output_task": "output",
                                "raw_input_task": [
                                    {
                                        "content": "",
                                        "role": "system",
                                    },
                                    {
                                        "content": "test_2",
                                        "role": "user",
                                    },
                                ],
                            },
                        ],
                    },
                ],
            ),
        ],
    )
    def test_process(
        self,
        input: List[Dict[str, str]],
        group_generations: bool,
        expected: List[Dict[str, Any]],
    ) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=pipeline,
            group_generations=group_generations,
            num_generations=3,
        )
        result = next(task.process(input))
        assert result == expected

    def test_process_overriding_inputs(self) -> None:
        llm = DummyAsyncLLM()
        task = DummyTask(
            name="task",
            llm=llm,
            group_generations=False,
            num_generations=3,
            input_mappings={"instruction": "instruction_2"},
        )

        result = next(
            task.process_applying_mappings(
                [
                    {
                        "instruction": "instruction that won't be used but overriden by input mapping",
                        "instruction_2": "instruction that will be used as input",
                        "additional_info": "info",
                    }
                ]
            )
        )

        assert result == [
            {
                "additional_info": "info",
                "distilabel_metadata": {
                    "raw_input_task": [
                        {
                            "content": "",
                            "role": "system",
                        },
                        {
                            "content": "instruction that will be used as input",
                            "role": "user",
                        },
                    ],
                    "raw_output_task": "output",
                },
                "info_from_input": "info",
                "instruction": "instruction that won't be used but overriden by input mapping",
                "instruction_2": "instruction that will be used as input",
                "model_name": "test",
                "output": "output",
            },
            {
                "additional_info": "info",
                "distilabel_metadata": {
                    "raw_input_task": [
                        {
                            "content": "",
                            "role": "system",
                        },
                        {
                            "content": "instruction that will be used as input",
                            "role": "user",
                        },
                    ],
                    "raw_output_task": "output",
                },
                "info_from_input": "info",
                "instruction": "instruction that won't be used but overriden by input mapping",
                "instruction_2": "instruction that will be used as input",
                "model_name": "test",
                "output": "output",
            },
            {
                "additional_info": "info",
                "distilabel_metadata": {
                    "raw_input_task": [
                        {
                            "content": "",
                            "role": "system",
                        },
                        {
                            "content": "instruction that will be used as input",
                            "role": "user",
                        },
                    ],
                    "raw_output_task": "output",
                },
                "info_from_input": "info",
                "instruction": "instruction that won't be used but overriden by input mapping",
                "instruction_2": "instruction that will be used as input",
                "model_name": "test",
                "output": "output",
            },
        ]

    def test_process_with_runtime_parameters(self) -> None:
        # 1. Runtime parameters provided
        llm = DummyRuntimeLLM()  # type: ignore
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.set_runtime_parameters({"llm": {"runtime_parameter": 1}})
        assert task.load() is None
        assert task.llm.runtime_parameter == 1  # type: ignore
        assert task.llm.runtime_parameters_names == {
            "runtime_parameter": False,
            "runtime_parameter_optional": True,
            "generation_kwargs": {},
            "offline_batch_generation_block_until_done": True,
            "use_offline_batch_generation": True,
        }

        # 2. Runtime parameters in init
        llm = DummyRuntimeLLM(runtime_parameter=1)
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert task.load() is None
        assert task.llm.runtime_parameter == 1  # type: ignore
        assert task.llm.runtime_parameters_names == {
            "runtime_parameter": False,
            "runtime_parameter_optional": True,
            "generation_kwargs": {},
            "offline_batch_generation_block_until_done": True,
            "use_offline_batch_generation": True,
        }

        # 3. Runtime parameters in init superseded by runtime parameters
        llm = DummyRuntimeLLM(runtime_parameter=1)
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.set_runtime_parameters({"llm": {"runtime_parameter": 2}})
        assert task.load() is None
        assert task.llm.runtime_parameter == 2  # type: ignore
        assert task.llm.runtime_parameters_names == {
            "runtime_parameter": False,
            "runtime_parameter_optional": True,
            "generation_kwargs": {},
            "offline_batch_generation_block_until_done": True,
            "use_offline_batch_generation": True,
        }

    def test_serialization(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = DummyTask(name="task", llm=llm, pipeline=pipeline)
        assert task.dump() == {
            "name": "task",
            "add_raw_output": True,
            "add_raw_input": True,
            "input_mappings": {},
            "output_mappings": {},
            "inputs": ["instruction", "additional_info"],
            "outputs": ["output", "info_from_input"],
            "resources": {
                "cpus": None,
                "gpus": None,
                "memory": None,
                "replicas": 1,
                "resources": None,
            },
            "input_batch_size": 50,
            "llm": {
                "generation_kwargs": {},
                "structured_output": None,
                "jobs_ids": None,
                "offline_batch_generation_block_until_done": None,
                "use_offline_batch_generation": False,
                "type_info": {
                    "module": "tests.unit.conftest",
                    "name": "DummyAsyncLLM",
                },
            },
            "group_generations": False,
            "num_generations": 1,
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
                            "description": "The kwargs to be propagated to either `generate` or "
                            "`agenerate` methods within each `LLM`.",
                            "keys": [],
                            "name": "generation_kwargs",
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
                    "description": "The number of generations to be produced per input.",
                    "optional": True,
                },
            ],
            "use_cache": True,
            "type_info": {
                "module": "tests.unit.conftest",
                "name": "DummyTask",
            },
            "use_default_structured_output": False,
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_task = DummyTask.from_dict(task.dump())
            assert isinstance(new_task, DummyTask)

    @pytest.mark.parametrize(
        "add_raw_output, add_raw_input",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    def test_add_raw_input_and_or_output(
        self, add_raw_output: bool, add_raw_input: bool
    ) -> None:
        task = DummyTask(
            llm=DummyAsyncLLM(),
            add_raw_output=add_raw_output,
            add_raw_input=add_raw_input,
        )
        assert task.add_raw_output is add_raw_output
        assert task.add_raw_input is add_raw_input
        task.load()
        input = [
            {"instruction": "test_0", "additional_info": "additional_info_0"},
            {"instruction": "test_1", "additional_info": "additional_info_1"},
            {"instruction": "test_2", "additional_info": "additional_info_2"},
        ]
        result = next(task.process(input))
        import pprint

        pprint.pprint(result)

        if add_raw_output or add_raw_input:
            assert "distilabel_metadata" in result[0].keys()
            if add_raw_output:
                assert (
                    "raw_output_dummy_task_0" in result[0]["distilabel_metadata"].keys()
                )
            if add_raw_input:
                assert (
                    "raw_input_dummy_task_0" in result[0]["distilabel_metadata"].keys()
                )
        else:
            assert "distilabel_metadata" not in result[0].keys()
