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

from distilabel.steps.tasks.text_generation_with_image import TextGenerationWithImage
from tests.unit.conftest import DummyAsyncLLM

img_str = "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"
np.random.seed(42)
img_pil = Image.fromarray(
    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), "RGB"
)


class TestTextGenerationWithImage:
    def test_format_input(self) -> None:
        llm = DummyAsyncLLM()
        task = TextGenerationWithImage(llm=llm, image_type="url")
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
        task = TextGenerationWithImage(llm=llm, system_prompt="test", image_type="url")
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
        task = TextGenerationWithImage(llm=llm, image_type=image_type)
        task.load()
        result = next(task.process([{"instruction": "test", "image": image}]))

        assert result == [
            {
                "instruction": "test",
                "image": image,
                "generation": "output",
                "distilabel_metadata": {
                    "raw_output_text_generation_with_image_0": "output",
                    "raw_input_text_generation_with_image_0": [
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
                    "statistics_text_generation_with_image_0": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    },
                },
                "model_name": "test",
            }
        ]
