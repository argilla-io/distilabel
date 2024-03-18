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

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Union, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr, Field

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
        """Returns the model name used for the LLM."""
        pass

    @abstractmethod
    def generate(
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List[List[Union[str, None]]]:
        """Abstract method to be implemented by each LLM to generate `num_generations`
        per input in `inputs`."""
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

    _event_loop: "asyncio.AbstractEventLoop" = PrivateAttr(default=None)

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
    ) -> List[List[Union[str, None]]]:
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
        self.event_loop.close()


class CUDALLM(LLM):
    cuda_devices: Union[List[int], Literal["auto"]] = Field(default="auto")

    _llm_identifier: Union[str, None] = PrivateAttr(default=None)
    _device_llm_placement_map: Union[Dict[str, Any], None] = PrivateAttr(default=None)

    def load(self) -> None:
        # `_device_llm_placement_map` is not mandatory, but if it is provided, it will be
        # used to assign CUDA devices to the LLM.
        if self._device_llm_placement_map is not None:
            self._assign_cuda_devices()

        super().load()

    def set_device_placement_info(
        self, llm_identifier: str, device_llm_placement_map: Dict[str, Any]
    ) -> None:
        """Sets the value of `_device_llm_placement_map` to be used to assign CUDA devices
        to the LLM.

        Args:
            llm_identifier: the identifier of the LLM to be used as key in the device
                placement information.
            device_llm_placement_map: a dictionary with the device placement information for
                each LLM. It should have two keys. The first key is "lock" and its value is
                a lock object to be used to synchronize the access to the device placement
                information. The second key is "value" and its value is a dictionary with the
                device placement information for each LLM.
        """
        self._llm_identifier = llm_identifier
        self._device_llm_placement_map = device_llm_placement_map

    def _assign_cuda_devices(self) -> None:
        """Assigns CUDA devices to the LLM based on the device placement information provided
        in `_device_llm_placement_map`. If the `cuda_devices` attribute is set to "auto", it
        will be set to the first available CUDA device that is not going to be used by any
        other LLM. If the `cuda_devices` attribute is set to a list of devices, it will be
        checked if the devices are available to be used by the LLM. If not, a warning will be
        logged."""
        with self._device_llm_placement_map["lock"]:
            device_map = self._device_llm_placement_map["value"]

            if self.cuda_devices == "auto":
                self.cuda_devices.append(self._get_cuda_device())
            else:
                self._check_cuda_devices(device_map)

        self._set_cuda_visible_devices()

    def _check_cuda_devices(self, device_map: Dict[str, List[int]]) -> None:
        """Checks if the CUDA devices assigned to the LLM are also assigned to other LLMs.

        Args:
            device_map: a dictionary with the device placement information for each LLM.
        """
        for device in self.cuda_devices:
            for llm, devices in device_map.items():
                if device in devices:
                    self._logger.warning(
                        f"LLM with identifier '{llm}' is also going to use CUDA device "
                        f"'{device}'. This may lead to performance issues or runnning out"
                        " of memory depending on the device capabilities and the loaded"
                        " models."
                    )

    def _get_cuda_device(self, device_map: Dict[str, List[int]]) -> int:
        """Returns the first available CUDA device to be used by the LLM that is not going
        to be used by any other LLM.

        Args:
            device_map: a dictionary with the device placement information for each LLM.

        Returns:
            The first available CUDA device to be used by the LLM.

        Raises:
            RuntimeError: if there is no available CUDA device to be used by the LLM.
        """
        cuda_devices = self._get_cuda_devices()
        for device in cuda_devices:
            if all(device not in devices for devices in device_map.values()):
                return device

        raise RuntimeError(
            "Couldn't find an available CUDA device automatically to be used by the LLM"
            f" '{self._llm_identifier}'. For forcing the use of a specific device, set the"
            " `cuda_devices` attribute to a list with the desired device(s)."
        )

    def _set_cuda_visible_devices(self) -> None:
        """Sets the `CUDA_VISIBLE_DEVICES` environment variable to the list of CUDA devices
        to be used by the LLM.
        """
        if self.cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.devices)

    def _get_cuda_devices(self) -> List[int]:
        """Returns the list of available CUDA devices.

        Returns:
            The list with the ID of available CUDA devices.
        """
        import torch

        return list(range(torch.cuda.device_count()))
