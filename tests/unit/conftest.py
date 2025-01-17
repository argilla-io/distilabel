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

import atexit
import os
from typing import TYPE_CHECKING, Any, Dict, List, Union
from urllib.request import urlretrieve

import pytest
from pydantic import PrivateAttr

from distilabel.models.image_generation.base import AsyncImageGenerationModel
from distilabel.models.llms.base import LLM, AsyncLLM
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.typing import ChatType, FormattedInput, GenerateOutput


# Defined here too, so that the serde still works
class DummyAsyncLLM(AsyncLLM):
    structured_output: Any = None
    n_generations_supported: bool = True  # To work as OpenAI or an LLM that doesn't allow num_generations out of the box
    _num_generations_param_supported: bool = PrivateAttr(default=True)

    def load(self) -> None:
        self._num_generations_param_supported = self.n_generations_supported

    @property
    def model_name(self) -> str:
        return "test"

    async def agenerate(  # type: ignore
        self, input: "FormattedInput", num_generations: int = 1
    ) -> "GenerateOutput":
        return {
            "generations": ["output" for i in range(num_generations)],
            "statistics": {
                "input_tokens": [12] * num_generations,
                "output_tokens": [12] * num_generations,
            },
        }


class DummyLLM(LLM):
    structured_output: Any = None

    def load(self) -> None:
        super().load()

    @property
    def model_name(self) -> str:
        return "test"

    def generate(  # type: ignore
        self, inputs: "FormattedInput", num_generations: int = 1
    ) -> List["GenerateOutput"]:
        return [
            {
                "generations": [f"output {i}" for i in range(num_generations)],
                "statistics": {
                    "input_tokens": [12] * num_generations,
                    "output_tokens": [12] * num_generations,
                },
            }
        ] * len(inputs)


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
            {
                "generations": ["Hello Magpie"] * num_generations,
                "statistics": {
                    "input_tokens": [12] * num_generations,
                    "output_tokens": [12] * num_generations,
                },
            }
            for _ in range(len(inputs))
        ]


class DummyAsyncImageGenerationModel(AsyncImageGenerationModel):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "test"

    async def agenerate(  # type: ignore
        self, input: str, num_generations: int = 1
    ) -> list[dict[str, Any]]:
        import numpy as np
        from PIL import Image

        np.random.seed(42)
        arr = np.random.randint(0, 255, (100, 100, 3))
        random_image = Image.fromarray(arr, "RGB")
        from distilabel.models.image_generation.utils import image_to_str

        img_str = image_to_str(random_image)
        return [{"images": [img_str]} for _ in range(num_generations)]


class DummyTask(Task):
    @property
    def inputs(self) -> List[str]:
        return ["instruction", "additional_info"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": input["instruction"]},
        ]

    @property
    def outputs(self) -> List[str]:
        return ["output", "info_from_input"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        return {"output": output, "info_from_input": input["additional_info"]}  # type: ignore


class DummyTaskOfflineBatchGeneration(DummyTask):
    _can_be_used_with_offline_batch_generation = True


@pytest.fixture
def dummy_llm() -> AsyncLLM:
    return DummyAsyncLLM()


@pytest.fixture(scope="session")
def local_llamacpp_model_path(tmp_path_factory):
    """
    Session-scoped fixture that provides the local model path for LlamaCpp testing.

    Download a small test model to a temporary directory.
    The model is downloaded once per test session and cleaned up after all tests.

    Args:
        tmp_path_factory: Pytest fixture providing a temporary directory factory.

    Returns:
        str: The path to the local LlamaCpp model file.
    """
    model_name = "all-MiniLM-L6-v2-Q2_K.gguf"
    model_url = f"https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/{model_name}"
    tmp_path = tmp_path_factory.getbasetemp()
    model_path = tmp_path / model_name

    if not model_path.exists():
        urlretrieve(model_url, model_path)

    def cleanup():
        if model_path.exists():
            os.remove(model_path)

    # Register the cleanup function to be called at exit
    atexit.register(cleanup)

    return str(tmp_path)
