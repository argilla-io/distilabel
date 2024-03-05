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

import json
from typing import Any, Dict, Generator, List

from distilabel.pipeline.llm.base import LLM
from distilabel.pipeline.llm.openai import OpenAILLM
from distilabel.pipeline.local import Pipeline
from distilabel.pipeline.step.base import Step
from distilabel.pipeline.step.task.typing import ChatType
from distilabel.pipeline.step.typing import StepInput
from distilabel.utils.serialization import TYPE_INFO_KEY


class TestLLMSerialization:
    openai_llm_dump = {
        "model": "gpt-3.5-turbo",
        TYPE_INFO_KEY: {
            "module": "distilabel.pipeline.llm.openai",
            "name": "OpenAILLM",
        },
    }

    def test_openai_llm_dump(self):
        llm = OpenAILLM(api_key="api_key")
        assert llm.dump() == self.openai_llm_dump

    def test_openai_llm_from_dict(self):
        assert isinstance(OpenAILLM.from_dict(self.openai_llm_dump), OpenAILLM)


class Task(Step):
    llm: LLM

    def load(self) -> None:
        self.llm.load()

    @property
    def inputs(self) -> List[str]:
        return (
            list(self.input_mapping.values())
            if self.input_mapping is not None
            else ["instruction"]
        )

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": input[self.inputs[0]]},
        ]

    @property
    def outputs(self) -> List[str]:
        return (
            list(self.output_mapping.values())
            if self.output_mapping is not None
            else ["generation"]
        )

    def format_output(self, output: str) -> Dict[str, Any]:
        return {self.outputs[0]: output}

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        for input in inputs:
            formatted_input = self.format_input(input)
            output = self.llm.generate(formatted_input)
            formatted_output = self.format_output(output)
            input.update(formatted_output)
        yield inputs


class TestTaskSerialization:
    task_dump = json.loads(
        """{
    "name": "generate_response",
    "llm": {
        "model": "gpt-3.5-turbo",
        "type_info": {
        "module": "distilabel.pipeline.llm.openai",
        "name": "OpenAILLM"
        }
    },
    "input_batch_size": 50,
    "input_mappings": {},
    "output_mappings": {
        "generation": "output"
    },
    "runtime_parameters_info": [],
    "type_info": {
        "module": "tests.unit.pipeline.llm.test_serialization",
        "name": "Task"
    }
    }"""
    )

    def test_task_dump(self):
        with Pipeline():
            task = Task(
                name="generate_response",
                llm=OpenAILLM(api_key="sk-***"),
                output_mappings={"generation": "output"},
            )
            assert task.dump() == self.task_dump

    def test_openai_llm_from_dict(self):
        with Pipeline():
            assert isinstance(Task.from_dict(self.task_dump), Task)
