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

import multiprocessing as mp
import os
import sys
from typing import TYPE_CHECKING, Any, Generator, List, Union
from unittest import mock

import pytest
from distilabel.llms.base import LLM
from distilabel.llms.mixins import CudaDevicePlacementMixin

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


@pytest.fixture
def mock_pynvml() -> Generator[None, None, None]:
    """Mocks `pynvml` module and clears the environment variables before each test."""
    with mock.patch.dict(os.environ, clear=True):
        # Mock `pynvml` module to avoid installing it in the CI
        sys.modules["pynvml"] = mock.MagicMock()
        pynvml = sys.modules["pynvml"]
        pynvml.nvmlInit.return_value = 0
        pynvml.nvmlDeviceGetCount.return_value = 4
        yield


class DummyCudaLLM(LLM, CudaDevicePlacementMixin):
    def load(self) -> None:
        super().load()
        CudaDevicePlacementMixin.load(self)

    @property
    def model_name(self) -> str:
        return "test"

    def generate(
        self, inputs: List["ChatType"], num_generations: int = 1, **kwargs: Any
    ) -> List[List[Union[str, None]]]:
        return [["output" for _ in range(num_generations)] for _ in inputs]


@pytest.mark.usefixtures("mock_pynvml")
class TestCudaDevicePlacementMixin:
    def test_set_cuda_visible_devices(self) -> None:
        llm = DummyCudaLLM(cuda_devices=[0, 1])
        llm._llm_identifier = "unit-test"

        llm.load()

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_cuda_visible_devices_not_cuda_devices(self) -> None:
        llm = DummyCudaLLM()
        llm._llm_identifier = "unit-test"

        llm.load()

        assert os.getenv("CUDA_VISIBLE_DEVICES") is None

    def test_set_cuda_visible_devices_unvalid_devices(self) -> None:
        llm = DummyCudaLLM(cuda_devices=[5, 6])
        llm._llm_identifier = "unit-test"

        with pytest.raises(
            RuntimeError, match="Invalid CUDA devices for LLM 'unit-test'"
        ):
            llm.load()

    def test_set_device_placement_info(self) -> None:
        llm = DummyCudaLLM(cuda_devices="auto")

        with mp.Manager() as manager:
            llm.set_device_placement_info(
                llm_identifier="unit-test",
                device_llm_placement_map=manager.dict(),
                device_llm_placement_lock=manager.Lock(),  # type: ignore
            )

        assert llm._llm_identifier == "unit-test"
        assert llm._device_llm_placement_map is not None

    def test_set_cuda_visible_devices_auto(self) -> None:
        with mp.Manager() as manager:
            device_llm_placement_map = manager.dict()
            lock = manager.Lock()

            llm1 = DummyCudaLLM()
            llm1.set_device_placement_info(
                llm_identifier="unit-test-1",
                device_llm_placement_map=device_llm_placement_map,
                device_llm_placement_lock=lock,  # type: ignore
            )
            llm1.load()

            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

            llm2 = DummyCudaLLM()
            llm2.set_device_placement_info(
                llm_identifier="unit-test-2",
                device_llm_placement_map=device_llm_placement_map,
                device_llm_placement_lock=lock,  # type: ignore
            )
            llm2.load()

            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    def test_set_cuda_visible_devices_auto_not_enough_devices(self) -> None:
        with mp.Manager() as manager:
            device_llm_placement_map = manager.dict()
            lock = manager.Lock()

            with pytest.raises(
                RuntimeError, match="Couldn't find an available CUDA device"
            ):
                # 4 devices are available, but 5 LLMs are going to be loaded
                for i in range(5):
                    llm = DummyCudaLLM()
                    llm.set_device_placement_info(
                        llm_identifier=f"unit-test-{i}",
                        device_llm_placement_map=device_llm_placement_map,
                        device_llm_placement_lock=lock,  # type: ignore
                    )
                    llm.load()

    def test_check_cuda_devices(self, caplog) -> None:
        with mp.Manager() as manager:
            device_llm_placement_map = manager.dict()
            lock = manager.Lock()

            llm1 = DummyCudaLLM(cuda_devices=[1])
            llm1.set_device_placement_info(
                llm_identifier="unit-test-1",
                device_llm_placement_map=device_llm_placement_map,
                device_llm_placement_lock=lock,  # type: ignore
            )
            llm1.load()

            llm2 = DummyCudaLLM(cuda_devices=[1])
            llm2.set_device_placement_info(
                llm_identifier="unit-test-2",
                device_llm_placement_map=device_llm_placement_map,
                device_llm_placement_lock=lock,  # type: ignore
            )
            llm2.load()

            assert (
                "LLM with identifier 'unit-test-1' is also going to use CUDA device '1'"
                in caplog.text
            )
