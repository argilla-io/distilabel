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

from datasets import load_dataset

from distilabel.models.image_generation import InferenceEndpointsImageLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import ImageGeneration

ds = load_dataset("dvilasuero/finepersonas-v0.1-tiny", split="train").select(range(3))

with Pipeline(name="image_generation_pipeline") as pipeline:
    ilm = InferenceEndpointsImageLM(model_id="black-forest-labs/FLUX.1-schnell")

    img_generation = ImageGeneration(
        name="flux_schnell", llm=ilm, input_mappings={"prompt": "persona"}
    )

    keep_columns = KeepColumns(columns=["persona", "model_name", "image"])

    img_generation >> keep_columns


if __name__ == "__main__":
    dataset = pipeline.run(use_cache=False, dataset=ds)
    dataset.push_to_hub("plaguss/test-finepersonas-v0.1-tiny-flux-schnell")
