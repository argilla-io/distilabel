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

import importlib.util
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    get_args,
)

from distilabel.errors import DistilabelUserError

if TYPE_CHECKING:
    import instructor
    from anthropic import AsyncAnthropic
    from cohere import AsyncClient as AsyncCohere
    from groq import AsyncGroq
    from mistralai.async_client import MistralAsyncClient
    from openai import AsyncAzureOpenAI, AsyncOpenAI


InstructorFrameworks = Literal[
    "openai", "azure_openai", "anthropic", "cohere", "groq", "litellm", "mistral"
]
"""Available frameworks for the structured output configuration with `instructor`. """

InstructorAvailableClients: TypeAlias = Union[
    "AsyncAnthropic",
    "AsyncAzureOpenAI",
    "AsyncCohere",
    "AsyncGroq",
    "AsyncOpenAI",
    "MistralAsyncClient",
]
"""Available clients that can be wrapped with `instructor`. """


def _client_patcher(framework: InstructorFrameworks) -> Tuple[Callable, str]:
    """Helper function to return the appropriate instructor client for the given framework.

    Args:
        framework: The framework to use for the instructor client.

    Raises:
        ValueError: If the framework is not one of the available frameworks.

    Returns:
        Tuple of Callable and string, with the builder of the client patch and the
            default mode to use.
    """
    import instructor

    if framework in {"openai", "azure_openai"}:
        patch = instructor.from_openai, instructor.Mode.TOOLS
    elif framework == "anthropic":
        patch = instructor.from_anthropic, instructor.Mode.ANTHROPIC_JSON
    elif framework == "litellm":
        patch = instructor.from_litellm, instructor.Mode.TOOLS
    elif framework == "mistral":
        patch = instructor.from_mistral, instructor.Mode.MISTRAL_TOOLS
    elif framework == "cohere":
        patch = instructor.from_cohere, instructor.Mode.COHERE_TOOLS
    elif framework == "groq":
        patch = instructor.from_groq, instructor.Mode.TOOLS
    else:
        raise ValueError(
            f"Invalid framework '{framework}'. Must be one of {get_args(InstructorFrameworks)}"
        )

    return patch


def prepare_instructor(
    client: InstructorAvailableClients,
    mode: Optional["instructor.Mode"] = None,
    framework: Optional[InstructorFrameworks] = None,
) -> "instructor.AsyncInstructor":
    """Wraps the given client with the instructor client for the given framework.

    Args:
        client: The client to wrap with the instructor client, corresponds to the internal
            client we wrap on `LLM`, and one of the implemented in `instructor`.
        mode: One of the `instructor.Mode` values. Defaults to None.
        framework: The framework corresponding to the client. Defaults to None.

    Raises:
        ImportError: If `instructor` is not installed.
        ValueError: If the mode is not one of the available modes.

    Returns:
        patched_client: The instructor wrapping the original client to be used for
            structured generation.
    """
    if not importlib.util.find_spec("instructor"):
        raise ImportError(
            "`instructor` is not installed. Please install it using `pip install instructor`."
        )
    import instructor

    builder, default_mode = _client_patcher(framework)

    mode = mode or default_mode
    if mode.value not in [m.value for m in instructor.mode.Mode]:
        raise DistilabelUserError(
            f"Invalid mode '{mode}'. Must be one of {[m.value for m in instructor.mode.Mode]}",
            page="sections/how_to_guides/advanced/structured_generation/#instructor",
        )

    patched_client: instructor.AsyncInstructor = builder(client, mode=mode)

    return patched_client
