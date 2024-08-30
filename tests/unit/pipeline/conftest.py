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

from distilabel.pipeline.local import Pipeline

from .utils import DummyGeneratorStep, DummyGlobalStep, DummyStep1, DummyStep2


@pytest.fixture(name="pipeline")
def pipeline_fixture() -> Pipeline:
    return Pipeline(name="unit-test-pipeline")


@pytest.fixture(name="dummy_step_1")
def dummy_step_1_fixture(pipeline: "Pipeline") -> DummyStep1:
    return DummyStep1(name="dummy_step_1", pipeline=pipeline)


@pytest.fixture(name="dummy_step_2")
def dummy_step_2_fixture(pipeline: "Pipeline") -> DummyStep2:
    return DummyStep2(name="dummy_step_2", pipeline=pipeline)


@pytest.fixture(name="dummy_generator_step")
def dummy_generator_step_fixture(pipeline: "Pipeline") -> DummyGeneratorStep:
    return DummyGeneratorStep(name="dummy_generator_step", pipeline=pipeline)


@pytest.fixture(name="dummy_global_step")
def dummy_global_step_fixture(pipeline: "Pipeline") -> DummyGlobalStep:
    return DummyGlobalStep(name="dummy_global_step", pipeline=pipeline)
