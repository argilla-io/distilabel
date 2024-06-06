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

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

import pytest
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.base import (
    BasePipeline,
    _GlobalPipelineManager,
)
from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.write_buffer import _WriteBuffer
from distilabel.steps.base import Step, StepInput
from distilabel.steps.typing import StepOutput
from distilabel.utils.serialization import TYPE_INFO_KEY
from fsspec.implementations.local import LocalFileSystem
from pydantic import Field
from upath import UPath

from .utils import (
    DummyGeneratorStep,
    DummyGlobalStep,
    DummyStep1,
    DummyStep2,
)


class TestGlobalPipelineManager:
    def teardown_method(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)

    def test_set_pipeline(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline

    def test_set_pipeline_none(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)
        assert _GlobalPipelineManager.get_pipeline() is None

    def test_get_pipeline(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline


class TestBasePipeline:
    def test_context_manager(self) -> None:
        assert _GlobalPipelineManager.get_pipeline() is None

        with BasePipeline(name="unit-test-pipeline") as pipeline:
            assert pipeline is not None
            assert _GlobalPipelineManager.get_pipeline() == pipeline

        assert _GlobalPipelineManager.get_pipeline() is None

    @pytest.mark.parametrize("use_cache", [False, True])
    def test_load_batch_manager(self, use_cache: bool) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")
        pipeline._load_batch_manager(use_cache=True)
        pipeline._cache()

        with mock.patch(
            "distilabel.pipeline.base._BatchManager.load_from_cache"
        ) as mock_load_from_cache, mock.patch(
            "distilabel.pipeline.base._BatchManager.from_dag"
        ) as mock_from_dag:
            pipeline._load_batch_manager(use_cache=use_cache)

        if use_cache:
            mock_load_from_cache.assert_called_once_with(
                pipeline._cache_location["batch_manager"]
            )
            mock_from_dag.assert_not_called()
        else:
            mock_load_from_cache.assert_not_called()
            mock_from_dag.assert_called_once_with(pipeline.dag)

    def test_setup_write_buffer(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")

        pipeline._setup_write_buffer()
        assert isinstance(pipeline._write_buffer, _WriteBuffer)

    def test_set_logging_parameters(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")
        pipeline._set_logging_parameters({"unit-test": "yes"})

        assert pipeline._logging_parameters == {"unit-test": "yes"}

    def test_setup_fsspec(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")

        with mock.patch("fsspec.filesystem") as mock_filesystem:
            pipeline._setup_fsspec({"path": "gcs://my-bucket", "extra": "stuff"})

        mock_filesystem.assert_called_once_with("gcs", **{"extra": "stuff"})

    def test_setup_fsspec_default(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")
        pipeline._setup_fsspec()

        assert isinstance(pipeline._fs, LocalFileSystem)
        assert (
            pipeline._storage_base_path
            == f"file://{pipeline._cache_location['batch_input_data']}"
        )

    def test_setup_fsspec_raises_value_error(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")

        with pytest.raises(ValueError, match="The 'path' key must be present"):
            pipeline._setup_fsspec({"key": "random"})

    def test_send_batch_to_step(self) -> None:
        with BasePipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            global_step = DummyGlobalStep()

            generator >> [step, global_step]

        pipeline._batch_manager = mock.MagicMock()
        pipeline._setup_fsspec()

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            batch = _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
            pipeline._send_batch_to_step(batch)
            pipeline._batch_manager.set_last_batch_sent.assert_called_once_with(batch)

            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=step.name, last_batch=False)  # type: ignore
            )

        mock_write.assert_not_called()

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=global_step.name, last_batch=False)  # type: ignore
            )

        mock_write.assert_called_once_with(
            pipeline._fs,
            UPath(pipeline._storage_base_path) / global_step.name,
        )

        pipeline._use_fs_to_pass_data = True

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
            )

        mock_write.assert_not_called()

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=step.name, last_batch=False)  # type: ignore
            )
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=global_step.name, last_batch=False)  # type: ignore
            )

        mock_write.assert_has_calls(
            [
                mock.call(
                    pipeline._fs,
                    UPath(pipeline._storage_base_path) / step.name,
                ),
                mock.call(
                    pipeline._fs,
                    UPath(pipeline._storage_base_path) / global_step.name,
                ),
            ]
        )

    def test_get_runtime_parameters_info(self) -> None:
        class DummyStep1(Step):
            runtime_param1: RuntimeParameter[str] = Field(
                default=None, description="runtime_param1 description"
            )
            runtime_param2: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param2 description"
            )

            def process(self, inputs: StepInput) -> None:
                pass

        class DummyStep2(Step):
            runtime_param3: RuntimeParameter[str] = Field(
                default=None, description="runtime_param3 description"
            )
            runtime_param4: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param4 description"
            )

            def process(self, inputs: StepInput) -> None:
                pass

        with BasePipeline(name="unit-test-pipeline") as pipeline:
            DummyStep1(name="dummy_step_1")
            DummyStep2(name="dummy_step_2")

        assert pipeline.get_runtime_parameters_info() == {
            "dummy_step_1": [
                {
                    "description": "The number of rows that will contain the batches processed by the "
                    "step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
                {
                    "name": "runtime_param1",
                    "description": "runtime_param1 description",
                    "optional": False,
                },
                {
                    "name": "runtime_param2",
                    "description": "runtime_param2 description",
                    "optional": True,
                },
            ],
            "dummy_step_2": [
                {
                    "description": "The number of rows that will contain the batches processed by the "
                    "step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
                {
                    "name": "runtime_param3",
                    "description": "runtime_param3 description",
                    "optional": False,
                },
                {
                    "name": "runtime_param4",
                    "description": "runtime_param4 description",
                    "optional": True,
                },
            ],
        }

    # Test no log, Test log, test log without close match
    @pytest.mark.parametrize(
        "parameters, expected",
        (
            (
                {
                    "dummy_step_1": {"runtime_param1": "value1"},
                    "dummy_step_2": {"runtime_param3": "value1"},
                },
                "",
            ),
            (
                {
                    "dummy_step_1": {"runtime_param1": "value1"},
                    "dummy_step_2": {
                        "runtime_param3": "value1",
                        "runtime_param_unknown": "value1",
                    },
                },
                "Did you mean any of:",
            ),
            (
                {
                    "dummy_step_1": {"runtime_param1": "value1"},
                    "dummy_step_2": {
                        "runtime_param3": "value1",
                        "weird_name": "value1",
                    },
                },
                "Available runtime parameters for the step",
            ),
        ),
    )
    def test_check_runtime_parameters(
        self, caplog, parameters: Dict[str, Any], expected: str
    ) -> None:
        class DummyStep1(Step):
            runtime_param1: RuntimeParameter[str] = Field(
                default=None, description="runtime_param1 description"
            )
            runtime_param2: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param2 description"
            )

            def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
                yield [{}]

        class DummyStep2(Step):
            runtime_param3: RuntimeParameter[str] = Field(
                default=None, description="runtime_param3 description"
            )
            runtime_param4: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param4 description"
            )

            def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
                yield [{}]

        with BasePipeline(name="unit-test-pipeline") as pipeline:
            gen_step = DummyGeneratorStep(name="dummy_generator_step")
            step1 = DummyStep1(name="dummy_step_1")
            step2 = DummyStep2(name="dummy_step_2")

            gen_step >> step1 >> step2

        pipeline.run(parameters=parameters)
        if expected:
            assert expected in caplog.text
        else:
            assert "Did you mean any of:" not in expected
            assert "Available runtime parameters for the step" not in expected

    def test_cache_dir_env_variable(self) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["DISTILABEL_CACHE_DIR"] = "/tmp/unit-test"
            pipeline = BasePipeline(name="unit-test-pipeline")
            assert pipeline._cache_dir == Path("/tmp/unit-test")

    @pytest.mark.parametrize(
        "in_pipeline, names",
        (
            (
                True,
                [
                    "dummy_generator_step_0",
                    "dummy_step1_0",
                    "dummy_step2_0",
                    "dummy_step1_1",
                ],
            ),
            # TODO: Activate this test once we merge the option of not passing a Pipeline
            # (
            #     False, ["dummy_generator_step", "dummy_step1", "dummy_step2"]
            # )
        ),
    )
    def test_step_names_inferred(self, in_pipeline: bool, names: List[str]) -> None:
        if in_pipeline:
            with BasePipeline(name="unit-test-pipeline"):
                gen_step = DummyGeneratorStep()
                step1_0 = DummyStep1()
                step2 = DummyStep2()
                step1_1 = DummyStep1()

                gen_step >> step1_0 >> step2 >> step1_1
        else:
            gen_step = DummyGeneratorStep()
            step1_0 = DummyStep1()
            step2 = DummyStep2()
            step1_1 = DummyStep1()

        assert gen_step.name == names[0]
        assert step1_0.name == names[1]
        assert step2.name == names[2]
        assert step1_1.name == names[3]

    def test_infer_step_names_big_pipeline(self) -> None:
        # Tests that the name of the steps are inferred correctly when the pipeline is big (say 50 steps).
        with BasePipeline(name="unit-test-pipeline") as pipe:
            gen_step = DummyGeneratorStep()
            for _ in range(50):
                gen_step.connect(DummyStep1())
        assert list(pipe.dag.G)[-1] == "dummy_step1_49"


class TestPipelineSerialization:
    def test_base_pipeline_dump(self):
        pipeline = BasePipeline(name="unit-test-pipeline")
        dump = pipeline.dump()
        assert len(dump.keys()) == 2
        assert "pipeline" in dump
        assert "distilabel" in dump
        assert TYPE_INFO_KEY in dump["pipeline"]
        assert dump["pipeline"][TYPE_INFO_KEY]["module"] == "distilabel.pipeline.base"
        assert dump["pipeline"][TYPE_INFO_KEY]["name"] == "BasePipeline"

    def test_base_pipeline_from_dict(self):
        pipeline = BasePipeline(name="unit-test-pipeline")
        pipe = BasePipeline.from_dict(pipeline.dump())
        assert isinstance(pipe, BasePipeline)

    def test_pipeline_dump(self):
        from distilabel.pipeline.local import Pipeline

        pipeline = Pipeline(name="unit-test-pipeline")
        dump = pipeline.dump()
        assert len(dump.keys()) == 2
        assert "pipeline" in dump
        assert "distilabel" in dump
        assert TYPE_INFO_KEY in dump["pipeline"]
        assert dump["pipeline"][TYPE_INFO_KEY]["module"] == "distilabel.pipeline.local"
        assert dump["pipeline"][TYPE_INFO_KEY]["name"] == "Pipeline"

    @pytest.mark.parametrize(
        "format, name, loader",
        [
            ("yaml", "pipe.yaml", BasePipeline.from_yaml),
            ("json", "pipe.json", BasePipeline.from_json),
            ("invalid", "pipe.invalid", None),
        ],
    )
    def test_pipeline_to_from_file_format(
        self,
        format: str,
        name: str,
        loader: Callable,
    ) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = Path(tmpdirname) / name
            if format == "invalid":
                with pytest.raises(ValueError):
                    pipeline.save(filename, format=format)
            else:
                pipeline.save(filename, format=format)
                assert filename.exists()
                pipe_from_file = loader(filename)
                assert isinstance(pipe_from_file, BasePipeline)

    def test_base_pipeline_signature(self):
        pipeline = BasePipeline(name="unit-test-pipeline")
        # Doesn't matter if it's exactly this or not, the test should fail if we change the
        # way this is created.
        signature = pipeline._create_signature()
        assert signature == "da39a3ee5e6b4b0d3255bfef95601890afd80709"

        # Maybe not the best place for this test, but does the work for now
        from distilabel.pipeline.local import Pipeline
        from distilabel.pipeline.routing_batch_function import sample_n_steps

        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        sample_two_steps = sample_n_steps(2)

        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep()
            dummy_step_1_0 = DummyStep1()
            dummy_step_1_1 = DummyStep1()
            dummy_step_1_2 = DummyStep1()
            dummy_step_2 = DummyStep2()

            (
                dummy_generator
                >> sample_two_steps
                >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                >> dummy_step_2
            )

        signature = pipeline._create_signature()
        assert signature == "a11ac46253598e6fe126420b23b9ad31c6422c92"

    def test_binary_rshift_operator(self) -> None:
        # Tests the steps can be connected using the >> operator.
        from distilabel.pipeline.local import Pipeline

        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator.connect(dummy_step_1)
            dummy_step_1.connect(dummy_step_2)

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-3") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator >> dummy_step_1 >> dummy_step_2

            signature_2 = pipeline_2._create_signature()

        assert signature_1 == signature_2

    def test_binary_rshift_operator_with_list(self) -> None:
        # Tests the steps can be connected using the >> operator when using a list.
        from distilabel.pipeline.local import Pipeline

        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator.connect(dummy_step_1)
            dummy_generator.connect(dummy_step_2)

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator >> [dummy_step_1, dummy_step_2]

            signature_2 = pipeline_2._create_signature()

        assert signature_1 == signature_2

    def test_binary_rrshift_operator(self) -> None:
        # Tests that a list of steps can be connected to a single step using the >> operator.
        # It usses the __rrshift__ method instead of the __rshift__ as it applies to the list
        # instead of the Step.

        from distilabel.pipeline.local import Pipeline

        from tests.unit.pipeline.utils import DummyGlobalStep, DummyStep1, DummyStep2

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            dummy_step_1.connect(dummy_global)
            dummy_step_2.connect(dummy_global)

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            [dummy_step_1, dummy_step_2] >> dummy_global
            signature_2 = pipeline_2._create_signature()

        assert signature_1 == signature_2

    def test_binary_operators(self) -> None:
        # Tests the steps can be connected with the binary operators,
        # the general case of step1 >> [step2, step3] >> step4
        from distilabel.pipeline.local import Pipeline

        from tests.unit.pipeline.utils import (
            DummyGeneratorStep,
            DummyGlobalStep,
            DummyStep1,
            DummyStep2,
        )

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            dummy_generator.connect(dummy_step_1)
            dummy_generator.connect(dummy_step_2)
            dummy_step_1.connect(dummy_global)
            dummy_step_2.connect(dummy_global)

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            dummy_generator >> [dummy_step_1, dummy_step_2] >> dummy_global
            signature_2 = pipeline_2._create_signature()

        assert signature_1 == signature_2
