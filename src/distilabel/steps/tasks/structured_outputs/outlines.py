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
    Callable,
    Dict,
    Literal,
    Tuple,
    Type,
    get_args,
)

import pkg_resources
from pydantic import BaseModel

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.structured_outputs.utils import schema_as_dict

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import OutlinesStructuredOutputType

Frameworks = Literal["transformers", "llamacpp", "vllm"]

if importlib.util.find_spec("outlines"):
    outlines_below_0_1_0 = pkg_resources.parse_version(
        pkg_resources.get_distribution("outlines").version
    ) < pkg_resources.parse_version("0.1.0")
else:
    outlines_below_0_1_0 = True


def model_to_schema(schema: Type[BaseModel]) -> Dict[str, Any]:
    """Helper function to return a string representation of the schema from a `pydantic.BaseModel` class."""
    return json.dumps(schema.model_json_schema())


def _get_logits_processor(framework: Frameworks) -> Tuple[Callable, Callable]:
    """Helper function to return the appropriate logits processor for the given framework."""
    if framework not in Frameworks.__args__:
        raise DistilabelUserError(
            f"Invalid framework '{framework}'. Must be one of {get_args(Frameworks)}",
            page="sections/how_to_guides/advanced/structured_generation/",
        )

    if outlines_below_0_1_0:
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
            from outlines.integrations.vllm import (
                JSONLogitsProcessor,
                RegexLogitsProcessor,
            )

            return JSONLogitsProcessor, RegexLogitsProcessor
    else:
        from outlines.processors import JSONLogitsProcessor, RegexLogitsProcessor

        return JSONLogitsProcessor, RegexLogitsProcessor


def _get_outlines_tokenizer_or_model(llm: Any, framework: Frameworks) -> Callable:
    if outlines_below_0_1_0:
        return llm
    else:
        if framework == "llamacpp":
            from outlines.models.llamacpp import LlamaCppTokenizer

            return LlamaCppTokenizer(llm)
        elif framework == "transformers":
            from outlines.models.transformers import TransformerTokenizer

            return TransformerTokenizer(llm.tokenizer)
        elif framework == "vllm":
            return llm.get_tokenizer()


def prepare_guided_output(
    structured_output: "OutlinesStructuredOutputType",
    framework: Frameworks,
    llm: Any,
) -> Dict[str, Any]:
    """Prepares the `LLM` to generate guided output using `outlines`.

    It allows to generate JSON or Regex structured outputs for the integrated
    frameworks.

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

    if not importlib.util.find_spec("outlines"):
        raise ImportError(
            "Outlines is not installed. Please install it using `pip install outlines`."
        )

    json_processor, regex_processor = _get_logits_processor(framework)

    tokenizer_or_model = _get_outlines_tokenizer_or_model(llm, framework)

    format = structured_output.get("format")
    schema = structured_output.get("schema")

    assert schema is not None, "schema cannot be `None`"

    # If schema not informed (may be forgotten), try infering it
    if not format:
        if isinstance(schema, dict) or inspect.isclass(schema):
            format = "json"
        elif isinstance(schema, str):
            format = "regex"

    if format == "json":
        return {
            "processor": json_processor(
                schema,
                tokenizer_or_model,
                whitespace_pattern=structured_output.get("whitespace_pattern"),
            ),
            "schema": schema_as_dict(schema),
        }

    if format == "regex":
        return {"processor": regex_processor(schema, tokenizer_or_model)}

    raise DistilabelUserError(
        f"Invalid format '{format}'. Must be either 'json' or 'regex'.",
        page="sections/how_to_guides/advanced/structured_generation/",
    )
