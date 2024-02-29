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
from collections import defaultdict
from typing import Optional

from datasets import Dataset

from distilabel.pipeline.step.base import GlobalStep
from distilabel.pipeline.step.typing import StepInput, StepOutput


class PushToHub(GlobalStep):
    # NOTE: `process` should be able to not return anything i.e. LeafStep, or just return None
    def process(
        self,
        inputs: StepInput,
        repo_id: str,
        split: str = "train",
        private: bool = False,
        token: Optional[str] = None,
    ) -> StepOutput:
        dataset_dict = defaultdict(list)
        for input in inputs:
            for key, value in input.items():
                dataset_dict[key].append(value)
        dataset_dict = dict(dataset_dict)
        dataset = Dataset.from_dict(dataset_dict)
        dataset.push_to_hub(
            repo_id, split=split, private=private, token=token or os.getenv("HF_TOKEN")
        )
        yield [{}]
