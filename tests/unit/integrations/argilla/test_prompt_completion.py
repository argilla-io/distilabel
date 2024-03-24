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
from unittest.mock import patch

import argilla as rg
from distilabel.integrations.argilla.prompt_completion import PromptCompletionToArgilla
from distilabel.pipeline.local import Pipeline

MockFeedbackDataset = rg.FeedbackDataset(
    fields=[rg.TextField(name="prompt"), rg.TextField(name="completion")],  # type: ignore
    questions=[
        rg.LabelQuestion(  # type: ignore
            name="quality",
            title="What's the quality of the completion for the given prompt?",
            labels=["bad", "good", "excellent"],
        )
    ],
)


class TestPromptCompletionToArgilla:
    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        step = PromptCompletionToArgilla(
            name="step",
            api_url="https://example.com",
            api_key="api.key",  # type: ignore
            dataset_name="argilla",
            dataset_workspace="argilla",
            pipeline=pipeline,
        )
        with patch.object(PromptCompletionToArgilla, "load"):
            step.load()
        step._prompt = "prompt"
        step._completion = "completion"
        step._rg_dataset = MockFeedbackDataset  # type: ignore

        assert list(step.process([{"prompt": "test", "completion": "test"}])) == [[{}]]
        assert len(step._rg_dataset.records) == 1

    def test_serialization(self) -> None:
        os.environ["ARGILLA_API_KEY"] = "api.key"

        pipeline = Pipeline(name="unit-test-pipeline")
        step = PromptCompletionToArgilla(
            name="step",
            api_url="https://example.com",
            dataset_name="argilla",
            dataset_workspace="argilla",
            pipeline=pipeline,
        )
        assert step.dump() == {
            "name": "step",
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "api_url": "https://example.com",
            "dataset_name": "argilla",
            "dataset_workspace": "argilla",
            "runtime_parameters_info": [],
            "type_info": {
                "module": "distilabel.integrations.argilla.prompt_completion",
                "name": "PromptCompletionToArgilla",
            },
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_step = PromptCompletionToArgilla.from_dict(step.dump())
            assert isinstance(new_step, PromptCompletionToArgilla)
