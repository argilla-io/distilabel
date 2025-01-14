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
    Union,
    get_args,
)

import pkg_resources
from pydantic import BaseModel

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.structured_outputs.utils import schema_as_dict

if TYPE_CHECKING:  # noqa
    from llama_cpp import Llama  # noqa
    from transformers import Pipeline  # noqa
    from vllm import LLM as _vLLM  # noqa

    from distilabel.typing import OutlinesStructuredOutputType  # noqa

Frameworks = Literal["transformers", "llamacpp", "vllm"]


def _is_outlines_version_below_0_1_0() -> bool:
    """Helper function to check outlines availability and version.

    Returns:
        bool: True if outlines is not installed or version is below 0.1.0
    """
    if not importlib.util.find_spec("outlines"):
        raise ImportError(
            "Outlines is not installed. Please install it using `pip install outlines`."
        )
    version = pkg_resources.get_distribution("outlines").version
    return pkg_resources.parse_version(version) < pkg_resources.parse_version("0.1.0")


def model_to_schema(schema: Type[BaseModel]) -> Dict[str, Any]:
    """Helper function to return a string representation of the schema from a `pydantic.BaseModel` class."""
    return json.dumps(schema.model_json_schema())


def _get_logits_processor(framework: Frameworks) -> Tuple[Callable, Callable]:
    """Helper function to return the appropriate logits processors for the given framework."""
    if _is_outlines_version_below_0_1_0():
        processors = {
            "transformers": (
                "outlines.integrations.transformers",
                "JSONPrefixAllowedTokens",
                "RegexPrefixAllowedTokens",
            ),
            "llamacpp": (
                "outlines.integrations.llamacpp",
                "JSONLogitsProcessor",
                "RegexLogitsProcessor",
            ),
            "vllm": (
                "outlines.integrations.vllm",
                "JSONLogitsProcessor",
                "RegexLogitsProcessor",
            ),
        }
    else:
        processors = {
            "transformers": (
                "outlines.processors",
                "JSONLogitsProcessor",
                "RegexLogitsProcessor",
            ),
            "llamacpp": (
                "outlines.processors",
                "JSONLogitsProcessor",
                "RegexLogitsProcessor",
            ),
            "vllm": (
                "outlines.processors",
                "JSONLogitsProcessor",
                "RegexLogitsProcessor",
            ),
        }

    if framework not in processors:
        raise DistilabelUserError(
            f"Invalid framework '{framework}'. Must be one of {get_args(Frameworks)}",
            page="sections/how_to_guides/advanced/structured_generation/",
        )

    module_path, json_cls, regex_cls = processors[framework]
    module = importlib.import_module(module_path)
    return getattr(module, json_cls), getattr(module, regex_cls)


def _get_tokenizer_from_model(
    llm: Union["_vLLM", "Pipeline", "Llama"],
    framework: Frameworks,
) -> Callable:
    if framework == "llamacpp":
        from outlines.models.llamacpp import LlamaCppTokenizer

        return LlamaCppTokenizer(llm)
    if framework == "transformers":
        from outlines.models.transformers import TransformerTokenizer

        return TransformerTokenizer(llm.tokenizer)
    if framework == "vllm":
        from outlines.models.vllm import adapt_tokenizer

        return adapt_tokenizer(llm.get_tokenizer())


def prepare_guided_output(
    structured_output: "OutlinesStructuredOutputType",
    framework: Frameworks,
    llm: Union["_vLLM", "Pipeline", "Llama"],
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

    json_processor, regex_processor = _get_logits_processor(framework)

    format = structured_output.get("format")
    schema = structured_output.get("schema")

    assert schema is not None, "schema cannot be `None`"

    # If schema not informed (may be forgotten), try infering it
    if not format:
        if isinstance(schema, dict) or inspect.isclass(schema):
            format = "json"
        elif isinstance(schema, str):
            format = "regex"

    if _is_outlines_version_below_0_1_0():
        # use the llm for processor initialization
        model = llm
        tokenizer = None
    else:
        # use the tokenizer for processor initialization
        model = None
        tokenizer = _get_tokenizer_from_model(llm, framework)

    if format == "json":
        return {
            "processor": json_processor(
                schema,
                model or tokenizer,
                whitespace_pattern=structured_output.get("whitespace_pattern"),
            ),
            "schema": schema_as_dict(schema),
        }

    if format == "regex":
        return {"processor": regex_processor(schema, model or tokenizer)}

    raise DistilabelUserError(
        f"Invalid format '{format}'. Must be either 'json' or 'regex'.",
        page="sections/how_to_guides/advanced/structured_generation/",
    )
