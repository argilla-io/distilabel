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
from distilabel.pipeline import Pipeline


def test_pipeline_errors_if_generator_wrong_instance() -> None:
    with pytest.raises(
        ValueError,
        match="`generator` must be an instance of `LLM`, `ProcessLLM` or `LLMPool`",
    ):
        Pipeline(generator=2)


def test_pipeline_errors_if_labeller_wrong_instance() -> None:
    with pytest.raises(
        ValueError, match="`labeller` must be an instance of `LLM` or `ProcessLLM`"
    ):
        Pipeline(labeller=2)


def test_pipeline_errors_if_no_generator_and_labeller() -> None:
    with pytest.raises(
        ValueError, match="Either `generator` or `labeller` must be provided."
    ):
        Pipeline()
