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

from pydantic import PrivateAttr

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.pipeline.batch import _Batch


class DummyStep(Step):
    attr: int = 5
    do_fail: bool = False
    _ctr: int = PrivateAttr(default=0)

    _random: str = PrivateAttr(default="")

    def load(self) -> None:
        super().load()
        self._random = str(uuid4())

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        for input in inputs:
            input["response"] = f"I don't know - {self._ctr} - {self._random}"
            self._ctr += 1

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
        outputs = []
        for input_a, input_b in zip(*inputs):
            output = {**input_a, **input_b}
            output["response"] = f"I don't know - {self._ctr}"
            self._ctr += 1
            outputs.append(output)
        yield outputs


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


def test_cache_with_step_cache_false() -> None:
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
                use_cache=False,
            )

            step_generator >> step_a >> step_b

        distiset_0 = pipeline.run()

        with mock.patch.object(
            pipeline, "_run_step", wraps=pipeline._run_step
        ) as run_step_spy:
            distiset_1 = pipeline.run()

        # check that only `step_b` has been executed
        assert run_step_spy.call_count == 1

        assert (
            distiset_0["default"]["train"].to_list()
            != distiset_1["default"]["train"].to_list()
        )


def test_cache_with_step_changing() -> None:
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

            step_generator >> step_a >> step_b

        distiset_0 = pipeline.run()

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
                attr=103401234,  # change attribute so step is not the same
                input_batch_size=10,
                input_mappings={"instruction": "response"},
                output_mappings={"response": "response_1"},
                do_fail=False,
                use_cache=True,
            )

            step_generator >> step_a >> step_b

        with mock.patch.object(
            pipeline, "_run_step", wraps=pipeline._run_step
        ) as run_step_spy:
            distiset_1 = pipeline.run()

        # check that only `step_b` has been executed
        assert run_step_spy.call_count == 1

        assert (
            distiset_0["default"]["train"].to_list()
            != distiset_1["default"]["train"].to_list()
        )


def test_cache_with_intermediate_step_cache_false() -> None:
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
                use_cache=False,
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

        with mock.patch.object(
            pipeline, "_run_step", wraps=pipeline._run_step
        ) as run_step_spy:
            distiset_1 = pipeline.run()

        # check that only `step_b` and `step_c` has been executed
        assert run_step_spy.call_count == 2

        assert (
            distiset_0["default"]["train"].to_list()
            != distiset_1["default"]["train"].to_list()
        )


def test_cache_adding_step() -> None:
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

            step_generator >> step_a >> step_b

        distiset_0 = pipeline.run()

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

        with mock.patch.object(
            pipeline, "_run_step", wraps=pipeline._run_step
        ) as run_step_spy:
            distiset_1 = pipeline.run()

        # check that only `step_c` has been executed
        assert run_step_spy.call_count == 1

        dict_0 = distiset_0["default"]["train"].to_dict()
        dict_1 = distiset_1["default"]["train"].to_dict()
        del dict_1["response_2"]
        assert dict_0 == dict_1


def test_cache_adding_step_with_multiple_predecessor() -> None:
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
                output_mappings={"response": "response_1"},
                use_cache=True,
            )
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                output_mappings={"response": "response_2"},
                do_fail=False,
                use_cache=True,
            )

            step_generator >> [step_a, step_b]

        distiset_0 = pipeline.run()

        with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline:
            initial_batch_size = 8
            step_generator = LoadDataFromDicts(
                data=[{"instruction": "some text"}] * initial_batch_size * 6,
                batch_size=initial_batch_size,
            )

            step_a = DummyStep(
                name="step_a",
                input_batch_size=4,
                output_mappings={"response": "response_1"},
                use_cache=True,
            )
            step_b = DummyStep(
                name="step_b",
                input_batch_size=10,
                output_mappings={"response": "response_2"},
                do_fail=False,
                use_cache=True,
            )
            step_c = DummyStep2(
                name="step_c",
                input_batch_size=12,
                output_mappings={"response": "response_3"},
                use_cache=True,
            )

            step_generator >> [step_a, step_b] >> step_c

        with mock.patch.object(
            pipeline, "_run_step", wraps=pipeline._run_step
        ) as run_step_spy:
            distiset_1 = pipeline.run()

        # check that only `step_c` has been executed
        assert run_step_spy.call_count == 1

        for row_1, row_0_a, row_0_b in zip(
            distiset_1["default"]["train"],
            distiset_0["step_a"]["train"],
            distiset_0["step_b"]["train"],
        ):
            assert row_1["response_1"] == row_0_a["response_1"]
            assert row_1["response_2"] == row_0_b["response_2"]


def test_cache_with_offset() -> None:
    use_cache_per_step = True
    do_fail = False
    with TemporaryDirectory() as tmp_dir:
        with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline_0:
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
            original_process_batch = pipeline_0._process_batch

        def _process_batch_wrapper(
            batch: "_Batch", send_last_batch_flag: bool = True
        ) -> None:
            if batch.step_name == step_b.name and batch.seq_no == 2:
                pipeline_0._stop_called = True
            original_process_batch(batch)

        # Run first time and stop the pipeline when specific batch received (simulate CTRL + C)
        with mock.patch.object(pipeline_0, "_process_batch", _process_batch_wrapper):
            distiset_0 = pipeline_0.run(use_cache=False)

        assert len(distiset_0["default"]["train"]) == 12

        with Pipeline(name="test_pipeline_caching", cache_dir=tmp_dir) as pipeline_1:
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

        distiset_1 = pipeline_1.run()

    assert len(distiset_1["default"]["train"]) == 48
