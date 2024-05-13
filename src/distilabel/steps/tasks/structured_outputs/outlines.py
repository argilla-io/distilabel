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
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
    get_args,
)

from pydantic import BaseModel

Frameworks = Literal["transformers", "llamacpp", "vllm"]
"""Available frameworks for the structured output configuration. """


class StructuredOutputType(TypedDict):
    """TypedDict to represent the structured output configuration from outlines.

    Attributes:
        format: one of "json" or "regex".
        schema: the schema to use for the structured output. If "json", it
            can be a pydantic.BaseModel class, or the schema as a string,
            as obtained from `model_to_schema(BaseModel)`, if "regex", it
            should be a regex pattern as a string.
        whitespace_patterm: if "json" corresponds to a string or a list of
            strings with a pattern (doesn't impact string literals).
            For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
    """

    format: Literal["json", "regex"]
    schema: Union[str, Type[BaseModel]]
    whitespace_pattern: Optional[Union[str, List[str]]]


def model_to_schema(schema: Type[BaseModel]) -> Dict[str, Any]:
    """Helper function to return a string representation of the schema from a `pydantic.BaseModel` class."""
    return json.dumps(schema.model_json_schema())


def _schema_as_dict(schema: Union[str, Type[BaseModel]]) -> Dict[str, Any]:
    """Helper function to obtain the schema and simplify serialization."""
    if type(schema) == type(BaseModel):
        return schema.model_json_schema()
    elif isinstance(schema, str):
        return json.loads(schema)
    return schema


def _get_logits_processor(framework: Frameworks) -> Tuple[Callable, Callable]:
    """Helper function to return the appropriate logits processor for the given framework."""
    if framework == "transformers":
        from outlines.integrations.transformers import (
            JSONPrefixAllowedTokens,
            RegexPrefixAllowedTokens,
        )

        return JSONPrefixAllowedTokens, RegexPrefixAllowedTokens

    if framework == "llamacpp":
        from outlines.integrations.llamacpp import (
            JSONLogitsProcessor,
            RegexLogitsProcessor,
        )

        return JSONLogitsProcessor, RegexLogitsProcessor

    if framework == "vllm":
        from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor

        return JSONLogitsProcessor, RegexLogitsProcessor

    raise ValueError(
        f"Invalid framework '{framework}'. Must be one of {get_args(Frameworks)}"
    )


def prepare_guided_output(
    structured_output: StructuredOutputType,
    framework: Frameworks,
    llm: Any,
) -> Dict[str, Union[Callable, None]]:
    """Prepares the `LLM` to generate guided output using `outlines`.

    It allows to generate JSON or Regex structured outputs for the integrated
    frameworks.

    Args:
        structured_output: the structured output configuration.
        framework: the framework to use for the structured output.
        llm: the `LLM` instance, each framework requires one thing so it should
            be obtained in the `LLM` itself.

    Raises:
        ValueError if the format is not "json" or "regex".

    Returns:
        A dictionary containing the processor to use for the guided output, and in
        case of "json" will also include the schema as a dict, to simplify serialization
        and deserialization.
    """
    if not importlib.util.find_spec("outlines"):
        raise ImportError(
            "Outlines is not installed. Please install it using `pip install outlines`."
        )

    json_processor, regex_processor = _get_logits_processor(framework)

    format = structured_output.get("format")
    schema = structured_output.get("schema")

    if format == "json":
        return {
            "processor": json_processor(
                schema,
                llm,
                whitespace_pattern=structured_output.get("whitespace_pattern"),
            ),
            "schema": _schema_as_dict(schema),
        }

    if format == "regex":
        return {"processor": regex_processor(schema, llm)}

    raise ValueError(f"Invalid format '{format}'. Must be either 'json' or 'regex'.")
