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

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class Checkpointer(Step):
    repo_id: str
    private: bool = True

    _data_dir: Optional[Path] = None
    _counter: int = 0

    def load(self) -> None:
        super().load()
        from distilabel.distiset import create_distiset

        self._create_distiset = create_distiset

    def process(self, *inputs: StepInput) -> "StepOutput":
        distiset = self._create_distiset(
            data_dir=self._data_dir,
        )
        if distiset:
            distiset.push_to_hub(
                repo_id=self.repo_id,
                private=self.private,
                commit_message=f"Checkpoint {self._counter}",
            )
            self._counter += 1

        yield from inputs
