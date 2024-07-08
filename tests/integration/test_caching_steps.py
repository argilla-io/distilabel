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

from typing import TYPE_CHECKING, Any, Dict, Generator, List

from distilabel.pipeline import Pipeline
from distilabel.steps import GeneratorStep
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps import GeneratorStepOutput


class StepGenerator(GeneratorStep):
    num_batches: int

    @property
    def outputs(self) -> List[str]:
        return ["instruction"]

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        for i in range(self.num_batches):
            yield (
                [{"instruction": "some text"} for _ in range(self.batch_size)],  # type: ignore
                i == self.num_batches - 1,
            )  # type: ignore


class DummyStep(Step):
    do_fail: bool = False

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        import time

        time.sleep(0.1)

        for input in inputs:
            input["response"] = "I don't know"

        if self.do_fail:
            raise ValueError("The step failed")
        yield inputs

    @property
    def outputs(self) -> List[str]:
        return ["response"]


def test_cached_steps() -> None:
    def run_pipeline(do_fail: bool = False, use_cache: bool = True):
        with Pipeline(name="test_pipeline_caching") as pipeline:
            step_generator = StepGenerator(num_batches=4, batch_size=8)

            step_a = DummyStep(name="step_a", input_batch_size=10)
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_1"},
                do_fail=do_fail,
            )
            step_c = DummyStep(
                name="step_c",
                input_batch_size=10,
                input_mappings={"instruction": "response_1"},
                output_mappings={"response": "response_2"},
            )

            step_generator >> step_a >> step_b >> step_c

        pipeline.run(use_cache=use_cache)

    run_pipeline(do_fail=False, use_cache=False)
    # After the second step is run, we should see that the pipeline is run only for steps b and c,
    # not the generator or step_1
    print("######" * 4)
    print("Rerun failed pipeline")
    print("######" * 4)
    # run_pipeline(do_fail=False, use_cache=True)


if __name__ == "__main__":
    test_cached_steps()
