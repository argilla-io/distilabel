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

import asyncio
import inspect
import logging
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from distilabel.mixins.runtime_parameters import (
    RuntimeParameter,
    RuntimeParametersMixin,
)
from distilabel.utils.docstring import parse_google_docstring
from distilabel.utils.notebook import in_notebook
from distilabel.utils.serialization import _Serializable

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput, HiddenState
    from distilabel.mixins.runtime_parameters import RuntimeParametersNames
    from distilabel.steps.tasks.structured_outputs.outlines import (
        OutlinesStructuredOutput,
    )
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.utils.docstring import Docstring

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


if in_notebook():
    import nest_asyncio

    nest_asyncio.apply()


class LLM(RuntimeParametersMixin, BaseModel, _Serializable, ABC):
    """Base class for `LLM`s to be used in `distilabel` framework.

    To implement an `LLM` subclass, you need to subclass this class and implement:
        - `load` method to load the `LLM` if needed. Don't forget to call `super().load()`,
            so the `_logger` attribute is initialized.
        - `model_name` property to return the model name used for the LLM.
        - `generate` method to generate `num_generations` per input in `inputs`.

    Attributes:
        generation_kwargs: the kwargs to be propagated to either `generate` or `agenerate`
            methods within each `LLM`.
        structured_output: a dictionary containing the structured output configuration or if more
            fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.
        _logger: the logger to be used for the `LLM`. It will be initialized when the `load`
            method is called.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        validate_default=True,
        validate_assignment=True,
        extra="forbid",
    )

    generation_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="The kwargs to be propagated to either `generate` or `agenerate`"
        " methods within each `LLM`.",
    )
    # TODO: how to properly annotate OutlinesStructuredOutput? it raises a pydantic error,
    # structured_output: Optional[Union[Dict[str, Any], "OutlinesStructuredOutput"]] = None
    structured_output: Optional[Union[Dict[str, Any], Any]] = None
    # _structured_generator: Optional["OutlinesStructuredOutput"] = PrivateAttr(None)
    _structured_generator: Optional[Any] = PrivateAttr(...)

    _logger: Union[logging.Logger, None] = PrivateAttr(...)

    def load(self) -> None:
        """Method to be called to initialize the `LLM`, its logger and optionally the structured output generator."""
        self._logger = logging.getLogger(f"distilabel.llm.{self.model_name}")
        self._structured_generator = self._prepare_structured_output(
            self.structured_output
        )

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        pass

    @abstractmethod
    def generate(
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        """Abstract method to be implemented by each LLM to generate `num_generations`
        per input in `inputs`.

        Args:
            inputs: the list of inputs to generate responses for which follows OpenAI's
                API format:

                ```python
                [
                    {"role": "system", "content": "You're a helpful assistant..."},
                    {"role": "user", "content": "Give a template email for B2B communications..."},
                    {"role": "assistant", "content": "Sure, here's a template you can use..."},
                    {"role": "user", "content": "Modify the second paragraph..."}
                ]
                ```
            num_generations: the number of generations to generate per input.
            **kwargs: the additional kwargs to be used for the generation.
        """
        pass

    @property
    def generate_parameters(self) -> List["inspect.Parameter"]:
        """Returns the parameters of the `generate` method.

        Returns:
            A list containing the parameters of the `generate` method.
        """
        return list(inspect.signature(self.generate).parameters.values())

    @property
    def runtime_parameters_names(self) -> "RuntimeParametersNames":
        """Returns the runtime parameters of the `LLM`, which are combination of the
        attributes of the `LLM` type hinted with `RuntimeParameter` and the parameters
        of the `generate` method that are not `input` and `num_generations`.

        Returns:
            A dictionary with the name of the runtime parameters as keys and a boolean
            indicating if the parameter is optional or not.
        """
        runtime_parameters = super().runtime_parameters_names
        runtime_parameters["generation_kwargs"] = {}

        # runtime parameters from the `generate` method
        for param in self.generate_parameters:
            if param.name in ["input", "inputs", "num_generations"]:
                continue
            is_optional = param.default != inspect.Parameter.empty
            runtime_parameters["generation_kwargs"][param.name] = is_optional

        return runtime_parameters

    def get_runtime_parameters_info(self) -> List[Dict[str, Any]]:
        """Gets the information of the runtime parameters of the `LLM` such as the name
        and the description. This function is meant to include the information of the runtime
        parameters in the serialized data of the `LLM`.

        Returns:
            A list containing the information for each runtime parameter of the `LLM`.
        """
        runtime_parameters_info = super().get_runtime_parameters_info()

        generation_kwargs_info = next(
            runtime_parameter_info
            for runtime_parameter_info in runtime_parameters_info
            if runtime_parameter_info["name"] == "generation_kwargs"
        )

        generate_docstring_args = self.generate_parsed_docstring["args"]

        generation_kwargs_info["keys"] = []
        for key, value in generation_kwargs_info["optional"].items():
            info = {"name": key, "optional": value}
            if description := generate_docstring_args.get(key):
                info["description"] = description
            generation_kwargs_info["keys"].append(info)

        generation_kwargs_info.pop("optional")

        return runtime_parameters_info

    @cached_property
    def generate_parsed_docstring(self) -> "Docstring":
        """Returns the parsed docstring of the `generate` method.

        Returns:
            The parsed docstring of the `generate` method.
        """
        return parse_google_docstring(self.generate)

    def get_last_hidden_states(self, inputs: List["ChatType"]) -> List["HiddenState"]:
        """Method to get the last hidden states of the model for a list of inputs.

        Args:
            inputs: the list of inputs to get the last hidden states from.

        Returns:
            A list containing the last hidden state for each sequence using a NumPy array
                with shape [num_tokens, hidden_size].
        """
        raise NotImplementedError(
            f"Method `get_last_hidden_states` is not implemented for `{self.__class__.__name__}`"
        )

    def _prepare_structured_output(
        self, structured_output: Optional[Union[Dict[str, Any], Any]] = None
    ) -> Union["OutlinesStructuredOutput", None]:
        """Method to prepare the structured output generator from the given structured output variable.

        Args:
            structured_output: If given, it must be an instance of `OutlinesStructuredOutput`
                or a dictionary containing the "format", "structure" and optionally
                "whitespace_pattern" for "json" format. If `None` is given, it won't do anything.
                Defaults to `None`.

        Returns:
            `OutlinesStructuredOutput` instance to generate structured outputs via `LLM.generate` or `None`.
        """
        if structured_output:
            from distilabel.steps.tasks.structured_outputs.outlines import (
                OutlinesStructuredOutput,
            )

            if isinstance(structured_output, OutlinesStructuredOutput):
                return structured_output
            elif isinstance(structured_output, dict):
                # TODO: Document this argument could be a dictionary like:
                # {"format": "text" | "json" | "regex" | "cfg", "structure": Any}
                # Also "method": "outlines", could be included once we integrate other frameworks like `instructor`.
                output_format = structured_output.get("format", "text")
                output_structure = structured_output.get("structure")
                if (output_format == "json") and (not output_structure):
                    # If the dict contains json in the format but no output_structure, use this
                    # case: https://outlines-dev.github.io/outlines/reference/json_mode/
                    # which is equivalent to the JSON mode from OpenAI.
                    import outlines

                    output_structure = outlines.grammars.json
                    output_format = "cfg"

                return OutlinesStructuredOutput._from_llm(
                    self,
                    output_format=output_format,
                    output_structure=output_structure,
                    whitespace_pattern=structured_output.get("whitespace_pattern"),
                )
            self._logger.warning(
                "The structured output must be an instance of `OutlinesStructuredOutput` or a dictionary."
                " with the keys `format`, `structure` and optionally `whitespace_pattern`. The following won't"
                f" have any effect: {structured_output}"
            )

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        # Common dump from the LLM base model
        dump = super()._model_dump(obj, **kwargs)
        # Update the structured_output content
        if "structured_output" in dump.keys():
            dump["structured_output"] = self._structured_generator.dump()
        return dump

    @classmethod
    def from_dict(self, data: Dict[str, Any]) -> Self:
        """Loads the LLM from a dictionary and injects the structured output if found.

        Args:
            data: Serialized data of the LLM.

        Returns:
            The loaded LLM instance.
        """
        structured_output = data.pop("structured_output", None)
        # Load the LLM as usual and inject the structured output if found.
        llm = super().from_dict(data)
        if structured_output:
            from distilabel.steps.tasks.structured_outputs.outlines import (
                OutlinesStructuredOutput,
            )

            structured_output["llm"] = llm
            structured_output = OutlinesStructuredOutput.from_dict(structured_output)
            llm.structured_output = structured_output

        return llm


class AsyncLLM(LLM):
    """Abstract class for asynchronous LLMs, so as to benefit from the async capabilities
    of each LLM implementation. This class is meant to be subclassed by each LLM, and the
    method `agenerate` needs to be implemented to provide the asynchronous generation of
    responses.

    Attributes:
        _event_loop: the event loop to be used for the asynchronous generation of responses.
    """

    _event_loop: "asyncio.AbstractEventLoop" = PrivateAttr(default=None)

    @property
    def generate_parameters(self) -> List[inspect.Parameter]:
        """Returns the parameters of the `agenerate` method.

        Returns:
            A list containing the parameters of the `agenerate` method.
        """
        return list(inspect.signature(self.agenerate).parameters.values())

    @cached_property
    def generate_parsed_docstring(self) -> "Docstring":
        """Returns the parsed docstring of the `agenerate` method.

        Returns:
            The parsed docstring of the `agenerate` method.
        """
        return parse_google_docstring(self.agenerate)

    @property
    def event_loop(self) -> "asyncio.AbstractEventLoop":
        if self._event_loop is None:
            try:
                self._event_loop = asyncio.get_running_loop()
                if self._event_loop.is_closed():
                    self._event_loop = asyncio.new_event_loop()  # type: ignore
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    @abstractmethod
    async def agenerate(
        self, input: "ChatType", num_generations: int = 1, **kwargs: Any
    ) -> List[Union[str, None]]:
        """Method to generate a `num_generations` responses for a given input asynchronously,
        and executed concurrently in `generate` method.
        """
        pass

    def generate(
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        """Method to generate a list of responses asynchronously, returning the output
        synchronously awaiting for the response of each input sent to `agenerate`.
        """
        if self._structured_generator is not None:
            return self._structured_generator(  # type: ignore
                inputs,
                max_tokens=kwargs.get("max_new_tokens", None),
                stop_at=kwargs.get("stop", None),
            )

        async def agenerate(
            inputs: List["ChatType"], **kwargs: Any
        ) -> List[List[Union[str, None]]]:
            """Internal function to parallelize the asynchronous generation of responses."""
            tasks = [
                asyncio.create_task(
                    self.agenerate(
                        input=input, num_generations=num_generations, **kwargs
                    )
                )
                for input in inputs
            ]
            return await asyncio.gather(*tasks)

        return self.event_loop.run_until_complete(agenerate(inputs, **kwargs))

    def __del__(self) -> None:
        """Closes the event loop when the object is deleted."""
        if sys.meta_path is None:
            return
        if self.event_loop is not None:
            self.event_loop.close()
