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

from typing import TYPE_CHECKING, Any, List

from distilabel.models.llms.base import LLM, AsyncLLM
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin

if TYPE_CHECKING:
    from distilabel.models.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import FormattedInput


class DummyAsyncLLM(AsyncLLM):
    structured_output: Any = None

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "test"

    async def agenerate(  # type: ignore
        self, input: "FormattedInput", num_generations: int = 1
    ) -> "GenerateOutput":
        return ["output" for _ in range(num_generations)]


class DummySyncLLM(LLM):
    structured_output: Any = None

    def load(self) -> None:
        super().load()

    @property
    def model_name(self) -> str:
        return "test"

    def generate(  # type: ignore
        self, inputs: "FormattedInput", num_generations: int = 1
    ) -> "GenerateOutput":
        return [["output" for _ in range(num_generations)] for _ in range(len(inputs))]


class DummyMagpieLLM(LLM, MagpieChatTemplateMixin):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "test"

    def generate(
        self, inputs: List["FormattedInput"], num_generations: int = 1, **kwargs: Any
    ) -> List["GenerateOutput"]:
        return [
            ["Hello Magpie" for _ in range(num_generations)] for _ in range(len(inputs))
        ]
