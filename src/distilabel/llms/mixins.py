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

import json
import os
from contextlib import contextmanager
from typing import Dict, Generator, List, Literal, Union

import portalocker
from pydantic import BaseModel, Field, PrivateAttr


class CudaDevicePlacementMixin(BaseModel):
    """Mixin class to assign CUDA devices to the `LLM` based on the `cuda_devices` attribute
    and the device placement information provided in `_device_llm_placement_map`. Providing
    the device placement information is optional, but if it is provided, it will be used to
    assign CUDA devices to the `LLM`s, trying to avoid using the same device for different
    `LLM`s.

    Attributes:
        cuda_devices: a list with the ID of the CUDA devices to be used by the `LLM`. If set
            to "auto", the devices will be automatically assigned based on the device
            placement information provided in `_device_llm_placement_map`. If set to a list
            of devices, it will be checked if the devices are available to be used by the
            `LLM`. If not, a warning will be logged.
        _llm_identifier: the identifier of the `LLM` to be used as key in `_device_llm_placement_map`.
        _device_llm_placement_map: a dictionary with the device placement information for each
            `LLM`.
    """

    # TODO: this should be a runtime parameter
    cuda_devices: Union[List[int], Literal["auto"]] = Field(default="auto")

    _llm_identifier: Union[str, None] = PrivateAttr(default=None)
    _available_cuda_devices: List[int] = PrivateAttr(default_factory=list)
    _can_check_cuda_devices: bool = PrivateAttr(default=False)

    def load(self) -> None:
        """Assign CUDA devices to the LLM based on the device placement information provided
        in `_device_llm_placement_map`."""

        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self._available_cuda_devices = list(range(device_count))
            self._can_check_cuda_devices = True
        except ImportError as ie:
            if self.cuda_devices == "auto":
                raise ImportError(
                    "The 'pynvml' library is not installed. It is required to automatically"
                    " assign CUDA devices to the `LLM`s. Please, install it and try again."
                ) from ie

            if self.cuda_devices:
                self._logger.warning(  # type: ignore
                    "The 'pynvml' library is not installed. It is recommended to install it"
                    " to check if the CUDA devices assigned to the LLM are available."
                )

        self._assign_cuda_devices()

    @contextmanager
    def _device_llm_placement_map(self) -> Generator[Dict[str, List[int]], None, None]:
        """Reads the content of the device placement file of the node with a lock, yields
        the content, and writes the content back to the file after the context manager is
        closed. If the file doesn't exist, an empty dictionary will be yielded.

        Yields:
            The content of the device placement file.
        """
        content = {}
        try:
            file = "file.json"
            with open(file, "r+") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                if os.path.exists(file):
                    content = json.load(f)
                yield content
                f.seek(0)
                f.truncate()
                f.write(json.dumps(content))
        finally:
            if "f" in locals() and not f.closed:  # type: ignore
                f.close()  # type: ignore

    def _assign_cuda_devices(self) -> None:
        """Assigns CUDA devices to the LLM based on the device placement information provided
        in `_device_llm_placement_map`. If the `cuda_devices` attribute is set to "auto", it
        will be set to the first available CUDA device that is not going to be used by any
        other LLM. If the `cuda_devices` attribute is set to a list of devices, it will be
        checked if the devices are available to be used by the LLM. If not, a warning will be
        logged."""

        # Take the lock and read the device placement information for each LLM.
        with self._device_llm_placement_map() as device_map:
            if self.cuda_devices == "auto":
                self.cuda_devices = [self._get_cuda_device(device_map)]
            else:
                self._check_cuda_devices(device_map)

            device_map[self._llm_identifier] = self.cuda_devices  # type: ignore

        # `_device_llm_placement_map` was not provided and user didn't set the `cuda_devices`
        # attribute. In this case, the `cuda_devices` attribute will be set to an empty list.
        if self.cuda_devices == "auto":
            self.cuda_devices = []

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
                        f"'{device}'. This may lead to performance issues or running out"
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
        for device in self._available_cuda_devices:
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
        if not self.cuda_devices:
            return

        if self._can_check_cuda_devices and not all(
            device in self._available_cuda_devices for device in self.cuda_devices
        ):
            raise RuntimeError(
                f"Invalid CUDA devices for LLM '{self._llm_identifier}': {self.cuda_devices}."
                f" The available devices are: {self._available_cuda_devices}. Please, review"
                " the 'cuda_devices' attribute and try again."
            )

        cuda_devices = ",".join([str(device) for device in self.cuda_devices])
        self._logger.info(
            f"🎮 LLM '{self._llm_identifier}' is going to use the following CUDA devices:"
            f" {self.cuda_devices}."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
