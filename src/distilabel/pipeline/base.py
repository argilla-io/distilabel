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

from typing import TYPE_CHECKING

from distilabel.pipeline._dag import DAG

if TYPE_CHECKING:
    from distilabel.step.base import Step


class BasePipeline:
    def __init__(self) -> None:
        self.dag = DAG()

    def add_step(self, step: "Step", name: str) -> None:
        self.dag.add_step(step, name)

    def add_edge(self, from_step: str, to_step: str) -> None:
        self.dag.add_edge(from_step, to_step)
