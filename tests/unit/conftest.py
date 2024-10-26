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

from distilabel.models.llms.base import LLM, AsyncLLM
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.models.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import ChatType, FormattedInput


# Defined here too, so that the serde still works
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
        return [["output" for _ in range(num_generations)]]


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
