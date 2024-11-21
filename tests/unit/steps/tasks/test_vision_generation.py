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


from distilabel.steps.tasks.vision_generation import VisionGeneration
from tests.unit.conftest import DummyAsyncLLM


class TestVisionGeneration:
    def test_format_input(self) -> None:
        llm = DummyAsyncLLM()
        task = VisionGeneration(llm=llm, image_type="url")
        task.load()

        assert task.format_input({"instruction": "test", "image": "123kjh123"}) == [
            {
                "role": "user",
                "content": [
                    {"text": "test", "type": "text"},
                    {"type": "image_url", "image_url": {"url": "123kjh123"}},
                ],
            }
        ]

    def test_format_input_with_system_prompt(self) -> None:
        llm = DummyAsyncLLM()
        task = VisionGeneration(llm=llm, system_prompt="test", image_type="url")
        task.load()

        assert task.format_input({"instruction": "test", "image": "123kjh123"}) == [
            {"role": "system", "content": "test"},
            {
                "role": "user",
                "content": [
                    {"text": "test", "type": "text"},
                    {"type": "image_url", "image_url": {"url": "123kjh123"}},
                ],
            },
        ]

    def test_process(self) -> None:
        llm = DummyAsyncLLM()
        task = VisionGeneration(llm=llm, image_type="url")
        task.load()
        result = next(task.process([{"instruction": "test", "image": "123kjh123"}]))
        print(result)

        assert next(task.process([{"instruction": "test", "image": "123kjh123"}])) == [
            {
                "instruction": "test",
                "image": "123kjh123",
                "generation": "output",
                "distilabel_metadata": {
                    "raw_output_vision_generation_0": "output",
                    "raw_input_vision_generation_0": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "test"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "123kjh123"},
                                },
                            ],
                        }
                    ],
                    "statistics_vision_generation_0": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    },
                },
                "model_name": "test",
            }
        ]
