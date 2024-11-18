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

from abc import ABC, abstractmethod
from typing import Any

from distilabel.models.llms.base import LLM, AsyncLLM


class ImageGenerationModel(LLM, ABC):
    @abstractmethod
    def generate(
        self, input: str, num_generations: int = 1, **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        """Generates images from the provided input.

        Args:
            input: the prompt text to generate the image from.
            num_generations: the number of images to generate. Defaults to `1`.

        Returns:
            A list with a dictionary with the list of images generated.
        """
        pass


class AsyncImageGenerationModel(AsyncLLM, ABC):
    @abstractmethod
    async def agenerate(
        self, input: str, num_generations: int = 1, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Generates images from the provided input.

        Args:
            input: the input text to generate the image from.
            num_generations: the number of images to generate. Defaults to `1`.

        Returns:
            A list with a dictionary with the list of images generated.
        """
        pass
