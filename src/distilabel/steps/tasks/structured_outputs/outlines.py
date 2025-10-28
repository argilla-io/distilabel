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

import importlib
import importlib.util
import inspect
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Type,
    Union,
    get_args,
)

from pydantic import BaseModel

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.structured_outputs.utils import schema_as_dict

if TYPE_CHECKING:  # noqa
    from llama_cpp import Llama  # noqa
    from transformers import Pipeline  # noqa
    from vllm import LLM as _vLLM  # noqa
    import mlx.nn as nn  # noqa

    from distilabel.typing import OutlinesStructuredOutputType  # noqa

Frameworks = Literal["transformers", "llamacpp", "mlx"]


def _check_outlines_available() -> None:
    """Helper function to check outlines availability.

    Raises:
        ImportError: If outlines is not installed.
    """
    if not importlib.util.find_spec("outlines"):
        raise ImportError(
            "Outlines is not installed. Please install it using `pip install outlines`."
        )


def model_to_schema(schema: Type[BaseModel]) -> Dict[str, Any]:
    """Helper function to return a string representation of the schema from a `pydantic.BaseModel` class."""
    return json.dumps(schema.model_json_schema())


def _create_outlines_model(
    llm: Union["Pipeline", "Llama", "nn.Module"],
    framework: Frameworks,
) -> Any:
    """Create an outlines model wrapper for the given framework.

    Args:
        llm: The LLM instance.
        framework: The framework being used.

    Returns:
        The outlines model instance that can create logits processors.
    """
    _check_outlines_available()

    if framework not in get_args(Frameworks):
        raise DistilabelUserError(
            f"Invalid framework '{framework}'. Must be one of {get_args(Frameworks)}",
            page="sections/how_to_guides/advanced/structured_generation/",
        )
    if framework == "transformers":
        from outlines import from_transformers

        return from_transformers(llm.model, llm.tokenizer)
    elif framework == "llamacpp":
        from outlines import from_llamacpp

        return from_llamacpp(llm)
    elif framework == "mlx":
        from outlines import from_mlxlm

        return from_mlxlm(llm._model, llm._tokenizer)


def prepare_guided_output(
    structured_output: "OutlinesStructuredOutputType",
    framework: Frameworks,
    llm: Union["_vLLM", "Pipeline", "Llama"],
) -> Dict[str, Any]:
    """Prepares the `LLM` to generate guided output using `outlines` >= 1.2.6.

    It allows to generate JSON or Regex structured outputs for the integrated
    frameworks using the outlines model API.

    Args:
        structured_output: the structured output configuration.
        framework: the framework to use for the structured output.
        llm: the `LLM` instance, each framework requires one thing so it should
            be obtained in the `LLM` itself.

    Raises:
        ValueError: if the format is not "json" or "regex".

    Returns:
        A dictionary containing the processor to use for the guided output, and in
        case of "json" will also include the schema as a dict, to simplify serialization
        and deserialization.
    """

    format = structured_output.get("format")
    schema = structured_output.get("schema")

    assert schema is not None, "schema cannot be `None`"

    # If schema not informed (may be forgotten), try infering it
    if not format:
        if isinstance(schema, dict) or inspect.isclass(schema):
            format = "json"
        elif isinstance(schema, str):
            format = "regex"

    outlines_model = _create_outlines_model(llm, framework)
    if format == "json":
        from outlines.backends import get_json_schema_logits_processor

        if inspect.isclass(schema):
            schema_str = model_to_schema(schema)
        elif isinstance(schema, dict):
            schema_str = json.dumps(schema)
        else:
            schema_str = schema
        return {
            "processor": get_json_schema_logits_processor(
                None, outlines_model, schema_str
            ),
            "schema": schema_as_dict(schema),
        }

    if format == "regex":
        from outlines.backends import get_regex_logits_processor

        return {"processor": get_regex_logits_processor(None, outlines_model, schema)}

    raise DistilabelUserError(
        f"Invalid format '{format}'. Must be either 'json' or 'regex'.",
        page="sections/how_to_guides/advanced/structured_generation/",
    )
