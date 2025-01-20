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

from distilabel.models.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks.text_generation_with_image import TextGenerationWithImage

with Pipeline(name="vision_generation_pipeline") as pipeline:
    loader = LoadDataFromDicts(
        data=[
            {
                "instruction": "What’s in this image?",
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            }
        ],
    )

    llm = InferenceEndpointsLLM(
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    )

    vision = TextGenerationWithImage(name="vision_gen", llm=llm, image_type="url")

    loader >> vision


if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    distiset.push_to_hub("plaguss/test-vision-generation-Llama-3.2-11B-Vision-Instruct")
