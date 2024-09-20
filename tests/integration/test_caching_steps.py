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

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, Generator, List
from unittest import mock
from uuid import uuid4

import pytest
from pydantic import PrivateAttr

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.distiset import Distiset
    from distilabel.pipeline.batch import _Batch


class DummyStep(Step):
    do_fail: bool = False
    ctr: int = 0

    _random: str = PrivateAttr(default="")

    def load(self) -> None:
        super().load()
        self._random = str(uuid4())

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        for input in inputs:
            input["response"] = f"I don't know - {self.ctr} - {self._random}"
            self.ctr += 1

        if self.do_fail:
            raise ValueError("The step failed")
        yield inputs

    @property
    def outputs(self) -> List[str]:
        return ["response"]


class DummyStep2(DummyStep):
    def process(
        self, *inputs: StepInput
    ) -> Generator[List[Dict[str, Any]], None, None]:
        for input in inputs[0]:
            input["response"] = f"I don't know - {self.ctr}"
            self.ctr += 1
        yield inputs[0]


class OtherDummyStep(DummyStep):
    pass


def test_cache() -> None:
    with TemporaryDirectory() as tmp_dir:
        with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline:
            initial_batch_size = 8
            step_generator = LoadDataFromDicts(
                data=[{"instruction": "some text"}] * initial_batch_size * 6,
                batch_size=initial_batch_size,
            )

            step_a = DummyStep(
                name="step_a",
                input_batch_size=4,
                use_cache=True,
            )
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_1"},
                do_fail=False,
                use_cache=True,
            )
            step_c = DummyStep(
                name="step_c",
                input_batch_size=12,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_2"},
                use_cache=True,
            )

            step_generator >> step_a >> step_b >> step_c

    distiset_0 = pipeline.run()
    distiset_1 = pipeline.run()
    assert (
        distiset_0["default"]["train"].to_list()
        == distiset_1["default"]["train"].to_list()
    )

    distiset_2 = pipeline.run(use_cache=False)
    assert (
        distiset_0["default"]["train"].to_list()
        != distiset_2["default"]["train"].to_list()
    )


def test_cache_adding_step() -> None:
    with TemporaryDirectory() as tmp_dir:
        with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline1:
            initial_batch_size = 8
            step_generator = LoadDataFromDicts(
                data=[{"instruction": "some text"}] * initial_batch_size * 6,
                batch_size=initial_batch_size,
            )

            step_a = DummyStep(
                name="step_a",
                input_batch_size=4,
                use_cache=True,
            )
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_1"},
                do_fail=False,
                use_cache=True,
            )

            step_generator >> step_a >> step_b

        with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline2:
            initial_batch_size = 8
            step_generator = LoadDataFromDicts(
                data=[{"instruction": "some text"}] * initial_batch_size * 6,
                batch_size=initial_batch_size,
            )

            step_a = DummyStep(
                name="step_a",
                input_batch_size=4,
                use_cache=True,
            )
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_1"},
                do_fail=False,
                use_cache=True,
            )
            step_c = DummyStep(
                name="step_c",
                input_batch_size=12,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_2"},
                use_cache=True,
            )

            step_generator >> step_a >> step_b >> step_c

        # assert that only `step_c` has been executed
        pipeline1.run()
        # import pdb; pdb.set_trace()
        pipeline2.run()


def test_cached_steps() -> None:
    use_cache_per_step = True
    with TemporaryDirectory() as tmp_dir:
        from pathlib import Path

        tmp_dir = Path.home() / "Downloads/test_pipeline_caching"

        def run_pipeline(do_fail: bool = False, use_cache: bool = True):
            with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline:
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
                failed_distiset = pipeline.run(use_cache=use_cache)

            print("*****\nRun again\n*****")
            assert len(failed_distiset["default"]["train"]) == 24
            distiset = pipeline.run(use_cache=use_cache)
            # This is the dataset size that we should have after succeeding
            assert len(distiset["default"]["train"]) == 48

        run_pipeline(do_fail=False, use_cache=True)


def test_cached_steps_with_multiple_predecessors():
    use_cache_per_step = True
    with TemporaryDirectory() as tmp_dir:

        def run_pipeline(use_cache: bool = True):
            with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline:
                initial_batch_size = 8
                step_generator = LoadDataFromDicts(
                    data=[{"instruction": "some text"}] * initial_batch_size * 6,
                    batch_size=initial_batch_size,
                )

                a = DummyStep(
                    name="step_a", input_batch_size=4, use_cache=use_cache_per_step
                )
                b = DummyStep(
                    name="step_b",
                    input_batch_size=10,
                    output_mappings={"response": "response_1"},
                    use_cache=use_cache_per_step,
                )
                c = DummyStep2(
                    name="step_c",
                    input_batch_size=12,
                    input_mappings={"instruction": "response"},
                    output_mappings={"response": "response_2"},
                    use_cache=use_cache_per_step,
                )

                step_generator >> [a, b] >> c

                # Controlled failure of the Pipeline
                original_process_batch = pipeline._process_batch

                def _process_batch_wrapper(batch: "_Batch") -> None:
                    if batch.step_name == c.name and batch.seq_no == 1:
                        pipeline._stop_called = True
                    original_process_batch(batch)

            # Run first time and stop the pipeline when specific batch received (simulate CTRL + C)
            with mock.patch.object(pipeline, "_process_batch", _process_batch_wrapper):
                failed_distiset = pipeline.run(use_cache=use_cache)

            print("*****\nRun again\n*****")
            assert len(failed_distiset["default"]["train"]) == 36
            distiset = pipeline.run(use_cache=use_cache)
            # This is the dataset size that we should have after succeeding
            assert len(distiset["default"]["train"]) == 48

        run_pipeline(use_cache=True)


@pytest.mark.parametrize("with_successor", (True, False))
def test_cached_steps_changing(with_successor: bool) -> None:
    with TemporaryDirectory() as tmp_dir:

        def run_pipeline(
            step_b_name: str = "step_b", step_flag: bool = True
        ) -> "Distiset":
            # with Pipeline(name="test_pipeline_caching") as pipeline:
            with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline:
                initial_batch_size = 8
                step_generator = LoadDataFromDicts(
                    data=[{"instruction": "some text"}] * initial_batch_size * 6,
                    batch_size=initial_batch_size,
                )

                step_a = DummyStep(name="step_a", input_batch_size=4, use_cache=True)
                if step_flag:
                    step_b = DummyStep(
                        name=step_b_name,
                        input_batch_size=10,
                        input_mappings={"instruction": "response"},
                        output_mappings={"response": "response_1"},
                        use_cache=True,
                    )
                else:
                    step_b = OtherDummyStep(
                        name=step_b_name,
                        input_batch_size=10,
                        input_mappings={"instruction": "response"},
                        output_mappings={"response": "response_2"},
                        use_cache=True,
                    )

                step_generator >> step_a >> step_b

                if with_successor:
                    # TODO: Test including an extra step to check it
                    step_c = DummyStep(
                        name="step_c",
                        input_batch_size=12,
                        input_mappings={"instruction": "response"},
                        output_mappings={"response": "response_3"},
                        use_cache=True,
                    )
                    step_b >> step_c

            distiset = pipeline.run(use_cache=True)

            return distiset

        distiset_one = run_pipeline(step_flag=True)

        print("---\nRun again\n---")
        distiset_two = run_pipeline(step_flag=False)

        df1 = distiset_one["default"]["train"].to_pandas()
        df2 = distiset_two["default"]["train"].to_pandas()

        assert len(df1) == len(df2) == 48

        if not with_successor:
            assert df1.columns.to_list() == ["response", "response_1"]
            assert df2.columns.to_list() == ["response", "response_2"]
        else:
            assert df1.columns.to_list() == ["response", "response_1", "response_3"]
            assert df2.columns.to_list() == ["response", "response_2", "response_3"]


def test_use_cache_per_step() -> None:
    with TemporaryDirectory() as tmp_dir:

        def run_pipeline():
            with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline:
                initial_batch_size = 8
                step_generator = LoadDataFromDicts(
                    data=[{"instruction": "some text"}] * initial_batch_size * 6,
                    batch_size=initial_batch_size,
                )

                step_a = DummyStep(name="step_a", input_batch_size=4)
                step_b = DummyStep(
                    name="step_b",
                    input_batch_size=10,
                    input_mappings={"instruction": "response"},
                    use_cache=False,
                )
                step_c = DummyStep(
                    name="step_c",
                    input_batch_size=12,
                    input_mappings={"instruction": "response"},
                )

                step_generator >> step_a >> step_b >> step_c

            distiset = pipeline.run(use_cache=True)

            print("*****\nRun again\n*****")
            assert len(distiset["default"]["train"]) == 48
            distiset = pipeline.run(use_cache=True)
            # This is the dataset size that we should have after succeeding
            assert len(distiset["default"]["train"]) == 48

            # We test that even if the pipeline is the same, we are recomputing from the step b onwards,
            # by checking a specific message in the logs.
            msg2 = (
                "♻️ `_BatchManagerStep` for 'step_b' and successors will be recomputed"
            )
            with pipeline._cache_location["log_file"].open() as f:
                lines = "\n".join(f.readlines()[-50:])

            assert msg2 in lines

        run_pipeline()


# TODO: In a next step we have to include more tests (when the feature is developed)
# to check if a pipeline: a >> b >> c has a new step at the end: a >> b >> c >> d, and
# we can start from the previous pipeline. Currently this is not possible.

if __name__ == "__main__":
    # Used to run via python while manually testing, to inspect the logs
    # print("\n\n test_cached_steps \n\n")
    # test_cached_steps()
    # test_cache()
    test_cache_adding_step()

    # TODO: CONTINUE HERE, RUN THE STEP.
    # TODO: THE LAST MERGE FROM DEVELOP INTRODUCED A VARIABLE:
    # send_last_batch_flag WHICH ISN'T TRACKED
    # print("\n\n test_cached_steps_changing \n\n")
    # test_cached_steps_changing(False)
    # print("\n\n test_cached_steps_changing with successor \n\n")
    # test_cached_steps_changing(True)
    # print("\n\n test_cached_steps_with_multiple_predecessors \n\n")
    # test_cached_steps_with_multiple_predecessors()
    ## test_use_cache_per_step()
