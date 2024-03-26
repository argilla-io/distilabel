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

from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.text_generation import TextGeneration

from tests.unit.steps.task.utils import DummyLLM


class TestTextGeneration:
    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = TextGeneration(name="task", llm=llm, pipeline=pipeline)
        assert next(task.process([{"instruction": "test"}])) == [
            {"instruction": "test", "generation": "output", "model_name": "test"}
        ]
