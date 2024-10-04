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

from distilabel.llms.base import LLM, AsyncLLM
from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
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
        self, input: "FormattedInput", num_generations: int = 1
    ) -> "GenerateOutput":
        return ["output" for _ in range(num_generations)]


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

    The model path can be set using the LLAMACPP_TEST_MODEL_PATH environment variable.
    If not set, it downloads a small test model to a temporary directory.
    The model is downloaded once per test session and cleaned up after all tests.

    To use a custom model:
    1. Set the LLAMACPP_TEST_MODEL_PATH environment variable to the path of your model file.
    2. Ensure the model file exists at the specified path.

    Example:
        export LLAMACPP_TEST_MODEL_PATH="/path/to/your/model.gguf"

    Args:
        tmp_path_factory: Pytest fixture providing a temporary directory factory.

    Returns:
        str: The path to the local LlamaCpp model file.
    """
    print("\nLlamaCpp model path information:")

    # Check for environment variable first
    env_path = os.environ.get("LLAMACPP_TEST_MODEL_PATH")
    if env_path:
        print(f"Using custom model path from LLAMACPP_TEST_MODEL_PATH: {env_path}")
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                f"Custom model file not found at {env_path}. Please ensure the file exists."
            )
        return env_path

    print("LLAMACPP_TEST_MODEL_PATH not set. Using default test model.")
    print(
        "To use a custom model, set the LLAMACPP_TEST_MODEL_PATH environment variable to the path of your model file."
    )

    # If env var not set, use a small test model
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


def pytest_addoption(parser):
    """
    Add a command-line option to pytest for CPU-only testing.
    """
    parser.addoption(
        "--cpu-only", action="store", default=False, help="Run tests on CPU only"
    )


@pytest.fixture
def use_cpu(request):
    """
    Fixture to determine whether to use CPU based on command-line option.
    """
    return request.config.getoption("--cpu-only")
