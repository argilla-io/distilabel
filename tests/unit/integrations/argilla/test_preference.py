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
from distilabel.integrations.argilla.preference import PreferenceToArgilla
from distilabel.pipeline.local import Pipeline


def mock_feedback_dataset() -> rg.FeedbackDataset:
    return rg.FeedbackDataset(
        fields=[
            rg.TextField(name="id", title="id"),  # type: ignore
            rg.TextField(name="instruction", title="instruction"),  # type: ignore
            rg.TextField(name="generations-0", title="generations-0"),  # type: ignore
            rg.TextField(name="generations-1", title="generations-1"),  # type: ignore
        ],
        questions=[
            rg.RatingQuestion(  # type: ignore
                name="generations-0-rating",
                title="Rate generations-0 given instruction based on the annotation guidelines.",
                description=None,
                values=[1, 2, 3, 4, 5],
                required=True,
            ),
            rg.TextQuestion(  # type: ignore
                name="generations-0-rationale",
                title="Specify the rationale for generations-0's rating.",
                description=None,
                required=False,
            ),
            rg.RatingQuestion(  # type: ignore
                name="generations-1-rating",
                title="Rate generations-1 given instruction based on the annotation guidelines.",
                description="Ignore this question if the corresponding `generations-1` field is not available.",
                values=[1, 2, 3, 4, 5],
                required=False,
            ),
            rg.TextQuestion(  # type: ignore
                name="generations-1-rationale",
                title="Specify the rationale for generations-1's rating.",
                description="Ignore this question if the corresponding `generations-1` field is not available.",
                required=False,
            ),
        ],
    )


class TestPreferenceToArgilla:
    def test_process(self) -> None:
        pipeline = Pipeline()
        step = PreferenceToArgilla(
            name="step",
            num_generations=2,
            api_url="https://example.com",
            api_key="api.key",  # type: ignore
            dataset_name="argilla",
            dataset_workspace="argilla",
            pipeline=pipeline,
        )
        with patch.object(PreferenceToArgilla, "load"):
            step.load()
        step._instruction = "instruction"
        step._generations = "generations"
        step._rg_dataset = mock_feedback_dataset()  # type: ignore

        assert list(
            step.process([{"instruction": "test", "generations": ["test", "test"]}])
        ) == [[{"instruction": "test", "generations": ["test", "test"]}]]
        assert len(step._rg_dataset.records) == 1

    def test_serialization(self) -> None:
        os.environ["ARGILLA_API_KEY"] = "api.key"

        pipeline = Pipeline()
        step = PreferenceToArgilla(
            name="step",
            num_generations=2,
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
            "num_generations": 2,
            "dataset_name": "argilla",
            "dataset_workspace": "argilla",
            "api_url": "https://example.com",
            "runtime_parameters_info": [
                {
                    "name": "api_url",
                    "optional": True,
                    "description": "The base URL to use for the Argilla API requests.",
                },
                {
                    "name": "api_key",
                    "optional": True,
                    "description": "The API key to authenticate the requests to the Argilla API.",
                },
            ],
            "type_info": {
                "module": "distilabel.integrations.argilla.preference",
                "name": "PreferenceToArgilla",
            },
        }

        with Pipeline() as pipeline:
            new_step = PreferenceToArgilla.from_dict(step.dump())
            assert isinstance(new_step, PreferenceToArgilla)
