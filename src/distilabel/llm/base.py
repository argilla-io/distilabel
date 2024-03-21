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
from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict, PrivateAttr, SecretStr

from distilabel.utils.docstring import parse_google_docstring
from distilabel.utils.logging import get_logger
from distilabel.utils.serialization import _Serializable

if TYPE_CHECKING:
    from distilabel.llm.typing import GenerateOutput, HiddenState
    from distilabel.steps.task.typing import ChatType
    from distilabel.utils.docstring import Docstring


class LLM(BaseModel, _Serializable, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=(), validate_default=True
    )

    _values: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _logger: logging.Logger = PrivateAttr(get_logger("llm"))

    @abstractmethod
    def load(self) -> None:
        pass

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
        per input in `inputs`."""
        pass

    @cached_property
    def generate_parameters(self) -> List[inspect.Parameter]:
        """Returns the parameters of the `generate` method.

        Returns:
            A list containing the parameters of the `generate` method.
        """
        return list(inspect.signature(self.generate).parameters.values())

    @property
    def runtime_parameters_names(self) -> Dict[str, bool]:
        """Returns the runtime parameters of the `LLM`, which are combination of the
        attributes of the `LLM` type hinted with `RuntimeParameter` and the parameters
        of the `generate` method that are not `input` and `num_generations`.

        Returns:
            A dictionary with the name of the runtime parameters as keys and a boolean
            indicating if the parameter is optional or not.
        """
        runtime_parameters = {}
        for param in self.generate_parameters:
            if param.name not in ["input", "inputs", "num_generations"]:
                continue
            is_optional = param.default != inspect.Parameter.empty
            runtime_parameters[param.name] = is_optional
        return runtime_parameters

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

    def _handle_api_key_value(
        self,
        self_value: Union[str, SecretStr, None],
        load_value: Union[str, SecretStr, None],
        env_var: str,
    ) -> SecretStr:
        """Method to handle the API key for the LLM, either from the `self_value` or the
        `load_value` i.e. the value provided within the `__init__` method of the `LLM` if
        applicable, and the value provided via the `load` method as a `RuntimeParameter` propagated
        via the `llm_kwargs`. Additionally, the `env_var` is also provided to guide the user on
        what's the environment variable name that needs to be used to assign the API Key value.

        Args:
            self_value: the value provided within the `__init__` method of the `LLM`.
            load_value: the value provided via the `load` method as a `RuntimeParameter`.
            env_var: the environment variable name to be used to assign the API Key value.

        Raises:
            ValueError: if the `api_key` is not present in the `LLM`.
            ValueError: if the `api_key` is and is not provided either via the `api_key` arg or
                runtime parameter. At most one of them should be provided.

        Returns:
            The API key value as a `SecretStr`.
        """

        if not hasattr(self, "api_key"):
            raise ValueError(
                f"You are trying to assign the `api_key` to the current `LLM={self.__class__.__name__}`,"
                " but the `api_key` attribute is not present."
            )

        if self_value is None and load_value is None:
            raise ValueError(
                "You must provide an API key either via the `api_key` arg or runtime"
                f" parameter, or either via the `{env_var}` environment variable."
            )

        if self_value is not None and load_value is not None:
            raise ValueError(
                "You must provide an API key either via the `api_key` arg or runtime"
                f" parameter, or either via the `{env_var}` environment variable,"
                " but not both."
            )

        api_key = self_value if self_value is not None else load_value
        if isinstance(api_key, str):
            api_key = SecretStr(api_key)  # type: ignore
        return api_key  # type: ignore


class AsyncLLM(LLM):
    """Abstract class for asynchronous LLMs, so as to benefit from the async capabilities
    of each LLM implementation. This class is meant to be subclassed by each LLM, and the
    method `agenerate` needs to be implemented to provide the asynchronous generation of
    responses.
    """

    _event_loop: "asyncio.AbstractEventLoop" = PrivateAttr(default=None)

    @cached_property
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
        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()  # type: ignore
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
