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
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from distilabel.mixins.runtime_parameters import (
    RuntimeParameter,
    RuntimeParametersMixin,
)
from distilabel.utils.docstring import parse_google_docstring
from distilabel.utils.itertools import grouper
from distilabel.utils.serialization import _Serializable

if TYPE_CHECKING:
    from logging import Logger

    from distilabel.mixins.runtime_parameters import (
        RuntimeParameterInfo,
        RuntimeParametersNames,
    )
    from distilabel.utils.docstring import Docstring


class ImageGenerationModel(RuntimeParametersMixin, BaseModel, _Serializable, ABC):
    """Base class for `ImageGeneration` models.

    To implement an `ImageGeneration` subclass, you need to subclass this class and implement:
        - `load` method to load the `ImageGeneration` model if needed. Don't forget to call `super().load()`,
            so the `_logger` attribute is initialized.
        - `model_name` property to return the model name used for the LLM.
        - `generate` method to generate `num_generations` per input in `inputs`.

    Attributes:
        generation_kwargs: the kwargs to be propagated to either `generate` or `agenerate`
            methods within each `ImageGenerationModel`.
        _logger: the logger to be used for the `ImageGenerationModel`. It will be initialized
            when the `load` method is called.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        validate_default=True,
        validate_assignment=True,
        extra="forbid",
    )

    generation_kwargs: Optional[RuntimeParameter[dict[str, Any]]] = Field(
        default_factory=dict,
        description="The kwargs to be propagated to either `generate` or `agenerate`"
        " methods within each `ImageGenerationModel`.",
    )
    _logger: "Logger" = PrivateAttr(None)

    def load(self) -> None:
        """Method to be called to initialize the `ImageGenerationModel`, and its logger."""
        self._logger = logging.getLogger(
            f"distilabel.models.image_generation.{self.model_name}"
        )

    def unload(self) -> None:
        """Method to be called to unload the `ImageGenerationModel` and release any resources."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model name used for the `ImageGenerationModel`."""
        pass

    def get_generation_kwargs(self) -> dict[str, Any]:
        """Returns the generation kwargs to be used for the generation. This method can
        be overridden to provide a more complex logic for the generation kwargs.

        Returns:
            The kwargs to be used for the generation.
        """
        return self.generation_kwargs  # type: ignore

    @abstractmethod
    def generate(
        self, input: list[str], num_generations: int = 1, **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        """Generates images from the provided input.

        Args:
            input: the prompt text to generate the image from.
            num_generations: the number of images to generate. Defaults to `1`.

        Returns:
            A list with a dictionary with the list of images generated.
        """
        pass

    def generate_outputs(
        self,
        inputs: list[str],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """This method is defined for compatibility with the `LLMs`. It calls the `generate`
        method.
        """
        return self.generate(inputs=inputs, num_generations=num_generations, **kwargs)

    @property
    def generate_parameters(self) -> list["inspect.Parameter"]:
        """Returns the parameters of the `generate` method.

        Returns:
            A list containing the parameters of the `generate` method.
        """
        return list(inspect.signature(self.generate).parameters.values())

    @property
    def runtime_parameters_names(self) -> "RuntimeParametersNames":
        """Returns the runtime parameters of the `ImageGenerationModel`, which are combination of the
        attributes of the `ImageGenerationModel` type hinted with `RuntimeParameter` and the parameters
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

    def get_runtime_parameters_info(self) -> list["RuntimeParameterInfo"]:
        """Gets the information of the runtime parameters of the `ImageGenerationModel` such as the name
        and the description. This function is meant to include the information of the runtime
        parameters in the serialized data of the `ImageGenerationModel`.

        Returns:
            A list containing the information for each runtime parameter of the `ImageGenerationModel`.
        """
        runtime_parameters_info = super().get_runtime_parameters_info()

        generation_kwargs_info = next(
            (
                runtime_parameter_info
                for runtime_parameter_info in runtime_parameters_info
                if runtime_parameter_info["name"] == "generation_kwargs"
            ),
            None,
        )

        # If `generation_kwargs` attribute is present, we need to include the `generate`
        # method arguments as the information for this attribute.
        if generation_kwargs_info:
            generate_docstring_args = self.generate_parsed_docstring["args"]
            generation_kwargs_info["keys"] = []
            # TODO: This doesn't happen with LLM, but with ImageGenerationModel the optional key is not found
            # in a pipeline, due to some bug. For the moment this does the job. It may be
            # related to the InferenceEndpointsImageGeneration for example being both
            # ImageGenerationModel and InferenceEdnpointsLLM, but cannot find the point that makes the
            # error appear.
            if "optional" not in generation_kwargs_info:
                return runtime_parameters_info
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


class AsyncImageGenerationModel(ImageGenerationModel):
    """Abstract class for asynchronous `ImageGenerationModels`, to benefit from the async capabilities
    of each LLM implementation. This class is meant to be subclassed by each `ImageGenerationModel`, and the
    method `agenerate` needs to be implemented to provide the asynchronous generation of
    responses.

    Attributes:
        _event_loop: the event loop to be used for the asynchronous generation of responses.
    """

    _num_generations_param_supported = True
    _event_loop: "asyncio.AbstractEventLoop" = PrivateAttr(default=None)
    _new_event_loop: bool = PrivateAttr(default=False)

    @property
    def generate_parameters(self) -> list[inspect.Parameter]:
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
                    self._new_event_loop = True
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
                self._new_event_loop = True
        asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    @abstractmethod
    async def agenerate(
        self, input: str, num_generations: int = 1, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Generates images from the provided input.

        Args:
            input: the input text to generate the image from.
            num_generations: the number of images to generate. Defaults to `1`.

        Returns:
            A list with a dictionary with the list of images generated.
        """
        pass

    async def _agenerate(
        self, inputs: list[str], num_generations: int = 1, **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        """Internal function to concurrently generate images for a list of inputs.

        Args:
            inputs: the list of inputs to generate images for.
            num_generations: the number of generations to generate per input.
            **kwargs: the additional kwargs to be used for the generation.

        Returns:
            A list containing the generations for each input.
        """
        if self._num_generations_param_supported:
            tasks = [
                asyncio.create_task(
                    self.agenerate(
                        input=input, num_generations=num_generations, **kwargs
                    )
                )
                for input in inputs
            ]
            return await asyncio.gather(*tasks)

        tasks = [
            asyncio.create_task(self.agenerate(input=input, **kwargs))
            for input in inputs
            for _ in range(num_generations)
        ]
        outputs = [outputs[0] for outputs in await asyncio.gather(*tasks)]
        return [
            list(group)
            for group in grouper(outputs, n=num_generations, incomplete="ignore")
        ]

    def generate(
        self,
        inputs: list[str],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Method to generate a list of responses asynchronously, returning the output
        synchronously awaiting for the response of each input sent to `agenerate`.

        Args:
            inputs: the list of inputs to generate responses for.
            num_generations: the number of generations to generate per input.
            **kwargs: the additional kwargs to be used for the generation.

        Returns:
            A list containing the generations for each input.
        """
        return self.event_loop.run_until_complete(
            self._agenerate(inputs=inputs, num_generations=num_generations, **kwargs)
        )

    def __del__(self) -> None:
        """Closes the event loop when the object is deleted."""
        if sys.meta_path is None:
            return

        if self._new_event_loop:
            if self._event_loop.is_running():
                self._event_loop.stop()
            self._event_loop.close()
