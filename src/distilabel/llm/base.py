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
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, ConfigDict, PrivateAttr

from distilabel.utils.logging import get_logger
from distilabel.utils.serialization import _Serializable

if TYPE_CHECKING:
    from distilabel.llm.typing import HiddenState
    from distilabel.steps.task.typing import ChatType


class LLM(BaseModel, _Serializable, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    _values: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _logger: logging.Logger = PrivateAttr(get_logger("llm"))

    @abstractmethod
    def load(self) -> None:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    # TODO: update return type hint to `List[List[str]]` as for each input and depending
    # on the `num_generations` parameter we would like to return a list of responses
    # for each input.
    @abstractmethod
    def generate(
        self, inputs: List["ChatType"], *args: Any, **kwargs: Any
    ) -> List[str]:
        pass

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


class AsyncLLM(LLM):
    """Abstract class for asynchronous LLMs, so as to benefit from the async capabilities
    of each LLM implementation. This class is meant to be subclassed by each LLM, and the
    method `agenerate` needs to be implemented to provide the asynchronous generation of
    responses.
    """

    @abstractmethod
    async def agenerate(self, input: "ChatType", *args: Any, **kwargs: Any) -> str:
        """Method to generate a single response asynchronously, parallelized via `generate`."""
        pass

    def generate(
        self, inputs: List["ChatType"], *args: Any, **kwargs: Any
    ) -> List[str]:
        """Method to generate a list of responses asynchronously, returning the output
        synchronously awaiting for the response of each input sent to `agenerate`.
        """

        async def agenerate(
            inputs: List["ChatType"], *args: Any, **kwargs: Any
        ) -> List[str]:
            """Internal function to parallelize the asynchronous generation of responses."""
            tasks = [
                asyncio.create_task(self.agenerate(input, *args, **kwargs))
                for input in inputs
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(agenerate(inputs, *args, **kwargs))
