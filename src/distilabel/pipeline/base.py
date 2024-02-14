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

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from typing_extensions import Self

from distilabel.pipeline._dag import DAG

if TYPE_CHECKING:
    from distilabel.step.base import Step


class _GlobalPipelineManager:
    _context_global_pipeline: Union["BasePipeline", None] = None

    @classmethod
    def set_pipeline(cls, pipeline: Union["BasePipeline", None] = None) -> None:
        cls._context_global_pipeline = pipeline

    @classmethod
    def get_pipeline(cls) -> Union["BasePipeline", None]:
        return cls._context_global_pipeline


class BasePipeline:
    def __init__(self) -> None:
        self.dag = DAG()

    def __enter__(self) -> Self:
        _GlobalPipelineManager.set_pipeline(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _GlobalPipelineManager.set_pipeline(None)

    def _add_step(self, step: "Step") -> None:
        self.dag.add_step(step)

    def _add_edge(self, from_step: str, to_step: str) -> None:
        self.dag.add_edge(from_step, to_step)

    def run(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.dag.validate(runtime_parameters=parameters)

    def _get_step_runtime_params(
        self, step_name: str, configuration: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Union[Dict[str, Any], None]:
        if configuration is None:
            return None
        return configuration.get(step_name, None)
