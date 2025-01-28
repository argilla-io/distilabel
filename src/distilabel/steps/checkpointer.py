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
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepOutput

from huggingface_hub import HfApi


class Checkpointer(Step):
    repo_id: str
    private: bool = True

    _counter: int = 0

    def load(self) -> None:
        super().load()
        self._api = HfApi()  # TODO: Add token
        # Create the repo if it doesn't exist
        if not self._api.repo_exists(repo_id=self.repo_id, repo_type="dataset"):
            self._logger.info(f"Creating repo {self.repo_id}")
            self._api.create_repo(
                repo_id=self.repo_id, repo_type="dataset", private=self.private
            )

    def process(self, *inputs: StepInput) -> "StepOutput":
        for i, input in enumerate(inputs):
            # Each section of *inputs corresponds to a different configuration of the pipeline
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as temp_file:
                for item in input:
                    json_line = json.dumps(item, ensure_ascii=False)
                    temp_file.write(json_line + "\n")
            try:
                self._api.upload_file(
                    path_or_fileobj=temp_file.name,
                    path_in_repo=f"config-{i}/train-{str(self._counter).zfill(5)}.jsonl",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Checkpoint {i}-{self._counter}",
                )
                self._logger.info(f"Uploaded checkpoint {i}-{self._counter}")
            finally:
                Path(temp_file.name).unlink()
                self._counter += 1

        yield from inputs
