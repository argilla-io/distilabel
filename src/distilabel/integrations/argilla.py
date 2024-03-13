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
from typing import TYPE_CHECKING, Annotated, List, Optional, Union

from pydantic import Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import override

try:
    import argilla as rg
except ImportError as ie:
    raise ImportError(
        "Argilla is not installed. Please install it using `pip install argilla`."
    ) from ie

from distilabel.steps.base import Step

if TYPE_CHECKING:
    from argilla.client.feedback.dataset.remote.dataset import RemoteFeedbackDataset

    from distilabel.steps.base import StepInput
    from distilabel.steps.typing import StepOutput


class PromptCompletionToArgilla(Step):
    api_url: str
    api_key: Annotated[Optional[SecretStr], Field(validate_default=True)] = None

    dataset_name: str
    dataset_workspace: Optional[str] = None

    _rg_dataset: Optional["RemoteFeedbackDataset"] = PrivateAttr(...)

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_none(cls, v: Union[str, SecretStr, None]) -> SecretStr:
        """Ensures that either the `api_key` or the environment variable `ARGILLA_API_KEY` are set.

        Additionally, the `api_key` when provided is casted to `pydantic.SecretStr` to prevent it
        from being leaked and/or included within the logs or the serialization of the object.
        """
        v = v or os.getenv("ARGILLA_API_KEY", None)  # type: ignore
        if v is None:
            raise ValueError("You must provide an API key to use Argilla.")
        if not isinstance(v, SecretStr):
            v = SecretStr(v)
        return v

    def load(self) -> None:
        try:
            rg.init(api_url=self.api_url, api_key=self.api_key.get_secret_value())  # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to initialize the Argilla API: {e}") from e

        # TODO: shoudln't `input_mappings` always have a value? It would make things easier
        try:
            self._prompt = (
                self.input_mappings["prompt"] if self.input_mappings else "prompt"
            )
            self._completion = (
                self.input_mappings["completion"]
                if self.input_mappings
                else "completion"
            )

            _rg_dataset = rg.FeedbackDataset(
                fields=[
                    rg.TextField(name=self._prompt),
                    rg.TextField(name=self._completion),
                ],  # type: ignore
                questions=[
                    rg.LabelQuestion(  # type: ignore
                        name="quality",
                        title=f"What's the quality of the {self._completion} for the given {self._prompt}?",
                        labels=["bad", "good", "excellent"],
                    )
                ],
            )
            self._rg_dataset = _rg_dataset.push_to_argilla(
                name=self.dataset_name, workspace=self.dataset_workspace
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize the Argilla dataset: {e}") from e

    @property
    def inputs(self) -> List[str]:
        return ["prompt", "completion"]

    @property
    def outputs(self) -> List[str]:
        return []

    @override
    def process(self, inputs: "StepInput") -> "StepOutput":
        self._rg_dataset.add_records(  # type: ignore
            [
                rg.FeedbackRecord(
                    fields={
                        self._prompt: input["prompt"],
                        self._completion: input["completion"],
                    }
                )
                for input in inputs
            ]
        )
        yield [{}]
