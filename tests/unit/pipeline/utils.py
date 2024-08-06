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


from typing import List

from distilabel.pipeline.batch import _Batch
from distilabel.steps.base import GeneratorStep, GlobalStep, Step, StepInput
from distilabel.steps.typing import GeneratorStepOutput, StepOutput
from pydantic import Field


class DummyGeneratorStep(GeneratorStep):
    outputs: List[str] = Field(default=["instruction"])
    def process(self, offset: int = 0) -> GeneratorStepOutput:  # type: ignore
        yield [{"instruction": "Generate an email..."}], False

class DummyGlobalStep(GlobalStep):
    inputs: List[str] = Field(default=["instruction"])
    def process(self) -> StepOutput:  # type: ignore
        yield [{"instruction": "Generate an email..."}]


class DummyStep1(Step):
    inputs: List[str] = Field(default=["instruction"])
    outputs: List[str] = Field(default=["response"])
    def process(self, input: StepInput) -> StepOutput:  # type: ignore
        yield [{"response": "response1"}]

class DummyStep2(Step):
    inputs: List[str] = Field(default=["response"])
    outputs: List[str] = Field(default=["evol_response"])
    def process(self, *inputs: StepInput) -> StepOutput:  # type: ignore
        yield [{"response": "response1"}]


def batch_gen(
    step_name: str,
    seq_no: int = 0,
    last_batch: bool = False,
    col_name: str = "a",
    num_rows: int = 5,
) -> _Batch:
    return _Batch(
        seq_no=seq_no,
        step_name=step_name,
        last_batch=last_batch,
        data=[[{col_name: i} for i in range(num_rows)]],
    )
