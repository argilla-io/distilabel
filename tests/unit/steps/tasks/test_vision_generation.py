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

from typing import Union

import numpy as np
import pytest
from PIL import Image

from distilabel.steps.tasks.vision_generation import VisionGeneration
from tests.unit.conftest import DummyAsyncLLM

img_str = "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"
np.random.seed(42)
img_pil = Image.fromarray(np.random.randint(0, 255, (100, 100, 3)), "RGB")


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

    @pytest.mark.parametrize(
        "image_type, image, expected",
        [
            ("url", "123kjh123", "123kjh123"),
            ("base64", img_str, f"data:image/jpeg;base64,{img_str}"),
            ("PIL", img_pil, f"data:image/jpeg;base64,{img_str}"),
        ],
    )
    def test_process(
        self, image_type: str, image: Union[str, "Image.Image"], expected: str
    ) -> None:
        llm = DummyAsyncLLM()
        task = VisionGeneration(llm=llm, image_type=image_type)
        task.load()
        result = next(task.process([{"instruction": "test", "image": image}]))

        assert result == [
            {
                "instruction": "test",
                "image": image,
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
                                    "image_url": {"url": expected},
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
