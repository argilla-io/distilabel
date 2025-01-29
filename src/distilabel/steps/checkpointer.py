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
from typing import TYPE_CHECKING, Optional

from pydantic import PrivateAttr

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepOutput

from huggingface_hub import HfApi


class HuggingFaceHubCheckpointer(Step):
    """Special type of step that uploads the data to a Hugging Face Hub dataset.

    A `Step` that uploads the data to a Hugging Face Hub dataset. The data is uploaded in JSONL format
    in a specific Hugging Face Dataset, which can be different to the one where the main distiset
    pipeline is saved. The data is checked every `input_batch_size` inputs, and a new file is created
    in the `repo_id` repository. There will be different config files depending on the leaf steps
    as in the pipeline, and each file will be numbered sequentially. As there will be writes every
    `input_batch_size` inputs, it's advisable not to set a small number on this step, as that
    will slow down the process.

    Attributes:
        repo_id:
            The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
            `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
            of the logged-in user.
        private:
            Whether the dataset repository should be set to private or not. Only affects repository creation:
            a repository that already exists will not be affected by that parameter.
        token:
            An optional authentication token for the Hugging Face Hub. If no token is passed, will default
            to the token saved locally when logging in with `huggingface-cli login`. Will raise an error
            if no token is passed and the user is not logged-in.

    Categories:
        - helper

    Examples:
        Do checkpoints of the data generated in a Hugging Face Hub dataset:

        ```python
        from typing import TYPE_CHECKING
        from datasets import Dataset

        from distilabel.pipeline import Pipeline
        from distilabel.steps import HuggingFaceHubCheckpointer
        from distilabel.steps.base import Step, StepInput

        if TYPE_CHECKING:
            from distilabel.typing import StepOutput

        # Create a dummy dataset
        dataset = Dataset.from_dict({"instruction": ["tell me lies"] * 100})

        with Pipeline(name="pipeline-with-checkpoints") as pipeline:
            text_generation = TextGeneration(
                llm=InferenceEndpointsLLM(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                template="Follow the following instruction: {{ instruction }}"
            )
            checkpoint = HuggingFaceHubCheckpointer(
                repo_id="username/streaming_checkpoint",
                private=True,
                input_batch_size=50  # Will save write the data to the dataset every 50 inputs
            )
            text_generation >> checkpoint
        ```

    """

    repo_id: str
    private: bool = True
    token: Optional[str] = None

    _counter: int = PrivateAttr(0)

    def load(self) -> None:
        super().load()
        if self.token is None:
            from distilabel.utils.huggingface import get_hf_token

            self.token = get_hf_token(self.__class__.__name__, "token")

        self._api = HfApi(token=self.token)
        # Create the repo if it doesn't exist
        if not self._api.repo_exists(repo_id=self.repo_id, repo_type="dataset"):
            self._logger.info(f"Creating repo {self.repo_id}")
            self._api.create_repo(
                repo_id=self.repo_id, repo_type="dataset", private=self.private
            )

    def process(self, *inputs: StepInput) -> "StepOutput":
        for i, input in enumerate(inputs):
            # Each section of *inputs corresponds to a different configuration of the pipeline
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as temp_file:
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
                    self._logger.info(f"⬆️ Uploaded checkpoint {i}-{self._counter}")
                finally:
                    self._counter += 1

        yield from inputs
