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
    Type,
    TypedDict,
    get_args,
)

from pydantic import BaseModel

if TYPE_CHECKING:
    import instructor


Frameworks = Literal[
    "openai", "azure_openai", "anthropic", "cohere", "groq", "litellm", "mistral"
]
"""Available frameworks for the structured output configuration with `instructor`. """


class InstructorStructuredOutputType(TypedDict):
    """TypedDict to represent the structured output configuration from `instructor`."""

    schema: Type[BaseModel]
    """The schema to use for the structured output, a pydantic.BaseModel class. """
    mode: Optional["instructor.Mode"] = None
    """Number of times to reask the model in case of error, if not set will default to the model's default. """
    max_retries: int


def _client_patcher(framework: Frameworks) -> Tuple[Callable, str]:
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
        return instructor.from_openai, instructor.Mode.TOOLS
    elif framework == "azure_openai":
        return instructor.from_anthropic, instructor.Mode.ANTHROPIC_JSON
    elif framework == "litellm":
        return instructor.from_litellm, instructor.Mode.TOOLS
    elif framework == "mistral":
        return instructor.from_mistral, instructor.Mode.MISTRAL_TOOLS
    elif framework == "cohere":
        return instructor.from_cohere, instructor.Mode.COHERE_TOOLS
    elif framework == "groq":
        return instructor.from_groq, instructor.Mode.TOOLS

    raise ValueError(
        f"Invalid framework '{framework}'. Must be one of {get_args(Frameworks)}"
    )


def prepare_instructor(
    client, mode: Optional["instructor.Mode"] = None, framework: Frameworks = "openai"
):
    if not importlib.util.find_spec("instructor"):
        raise ImportError(
            "instructor is not installed. Please install it using `pip install instructor`."
        )
    import instructor

    builder, default_mode = _client_patcher(framework)

    mode = mode or default_mode
    if mode not in [m.value for m in instructor.mode.Mode]:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of {[m.value for m in instructor.mode.Mode]}"
        )

    patched_client: instructor.Instructor = builder(client, mode=mode)

    return patched_client
