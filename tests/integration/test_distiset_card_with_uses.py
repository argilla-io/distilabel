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

from distilabel.pipeline import Pipeline
from distilabel.steps import (
    FormatTextGenerationDPO,
    FormatTextGenerationSFT,
    LoadDataFromDicts,
)


def test_dataset_card() -> None:
    with Pipeline() as pipeline:
        data = LoadDataFromDicts(
            data=[
                {
                    "instruction": "What's 2+2?",
                    "generation": "4",
                    "generations": ["4", "5"],
                    "ratings": [1, 5],
                },
            ]
        )
        formatter = FormatTextGenerationSFT()
        formatter_dpo = FormatTextGenerationDPO()

        data >> formatter >> formatter_dpo

    distiset = pipeline.run(use_cache=False)
    disti_card = distiset._get_card("user/repo_id")
    # Check that the card has the expected content
    assert "## Uses\n\n### Supervised Fine-Tuning (SFT)" in str(disti_card)
    assert "### Direct Preference Optimization (DPO)" in str(disti_card)


if __name__ == "__main__":
    test_dataset_card()
