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
from unittest import mock

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.pipeline.batch import _Batch


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


def original_test_cached_steps() -> None:
    # TODO: Make other types of pipelines with a step sending to multiple steps in parallel (step_a >> [step_b, step_c])
    use_cache_per_step = True

    def run_pipeline(do_fail: bool = False, use_cache: bool = True):
        with Pipeline(name="test_pipeline_caching") as pipeline:
            initial_batch_size = 8
            step_generator = LoadDataFromDicts(
                data=[{"instruction": "some text"}] * initial_batch_size * 6,
                batch_size=initial_batch_size,
            )

            step_a = DummyStep(
                name="step_a", input_batch_size=4, use_cache=use_cache_per_step
            )
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_1"},
                do_fail=do_fail,
                use_cache=use_cache_per_step,
            )
            step_c = DummyStep(
                name="step_c",
                input_batch_size=12,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_2"},
                use_cache=use_cache_per_step,
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


def test_cached_steps() -> None:
    # TODO: Make other types of pipelines with a step sending to multiple steps in parallel (step_a >> [step_b, step_c])
    use_cache_per_step = True

    def run_pipeline(do_fail: bool = False, use_cache: bool = True):
        with Pipeline(name="test_pipeline_caching") as pipeline:
            initial_batch_size = 8
            step_generator = LoadDataFromDicts(
                data=[{"instruction": "some text"}] * initial_batch_size * 6,
                batch_size=initial_batch_size,
            )

            step_a = DummyStep(
                name="step_a", input_batch_size=4, use_cache=use_cache_per_step
            )
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_1"},
                do_fail=do_fail,
                use_cache=use_cache_per_step,
            )
            step_c = DummyStep(
                name="step_c",
                input_batch_size=12,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_2"},
                use_cache=use_cache_per_step,
            )

            step_generator >> step_a >> step_b >> step_c

            # Controlled failure of the Pipeline
            original_process_batch = pipeline._process_batch

            def _process_batch_wrapper(batch: "_Batch") -> None:
                if batch.step_name == step_b.name and batch.seq_no == 3:
                    pipeline._stop_called = True
                original_process_batch(batch)

        # Run first time and stop the pipeline when specific batch received (simulate CTRL + C)
        with mock.patch.object(pipeline, "_process_batch", _process_batch_wrapper):
            pipeline.run(use_cache=use_cache)

        print("*****\nRun again\n*****")
        # pipeline.run(use_cache=use_cache)

    run_pipeline(do_fail=False, use_cache=True)


if __name__ == "__main__":
    # original_test_cached_steps()
    test_cached_steps()
