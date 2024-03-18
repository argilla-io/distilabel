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
from distilabel.steps.task.base import DataTask
from pydantic import ValidationError


class TestDataTask:
    def test_required_data(self) -> None:
        pipeline = Pipeline()
        with pytest.raises(ValidationError):
            DataTask(name="task", pipeline=pipeline)

    def test_process(self) -> None:
        pipeline = Pipeline()
        data: list[dict[str, str]] = [{"instruction": "test"}] * 10
        task = DataTask(name="task", pipeline=pipeline, data=data)
        result = list(task.process())
        assert result == [([{"instruction": "test"}] * 10, True)]
