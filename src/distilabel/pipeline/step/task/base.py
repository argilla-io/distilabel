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

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List

from distilabel.pipeline.llm.base import LLM
from distilabel.pipeline.step.base import Step, StepInput
from distilabel.pipeline.step.task.types import ChatType


class Task(Step, ABC):
    llm: LLM

    def load(self) -> None:
        self.llm.load()  # type: ignore

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> ChatType:
        pass

    @abstractmethod
    def format_output(self, output: str) -> Dict[str, Any]:
        pass

    def process(self, inputs: StepInput) -> Iterator[List[Dict[str, Any]]]:
        for input in inputs:
            formatted_input = self.format_input(input)
            output = self.llm.generate(formatted_input)  # type: ignore
            formatted_output = self.format_output(output)  # type: ignore
            input.update(formatted_output)
        yield inputs  # type: ignore
