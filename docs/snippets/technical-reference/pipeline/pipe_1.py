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

from distilabel.llm import InferenceEndpointsLLM
from distilabel.pipeline import pipeline
from distilabel.tasks import TextGenerationTask

pipe = pipeline(
    "preference",
    "text-quality",
    generator=InferenceEndpointsLLM(
        endpoint_name=endpoint_name,
        endpoint_namespace=endpoint_namespace,
        token=token,
        task=TextGenerationTask(),
        max_new_tokens=512,
        do_sample=True,
        prompt_format="notus",
    ),
    max_new_tokens=256,
    num_threads=2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)
