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

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import MagpieGenerator, Task
from distilabel.llms import InferenceEndpointsLLM
from distilabel.llms.base import LLM

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = "You are a customer support agent for a phone company. \
    Your purpose is to assist customers with their phone-related issues, \
    but you are not very patient and tend to be a bit rude. User queries  \
    will be straightforward and clear, but you will respond in a somewhat \
    blunt and curt manner. Remember to keep your responses concise and to \
    the point. User queries are often about phone plans, billing, and \
    technical issues. Your responses should be direct and focus on resolving \
    the issue at hand, but with a slightly abrasive tone. User queries will be \
    concise and to the point, User queries are often about phone plans, billing, \
    and technical issues."


class SFTPipeline:

    def __init__(
        self,
        llm: LLM,
        hf_token=None,
        generation_kwargs=None,
        n_turns: int = 1,
        num_rows: int = 10,
        batch_size: int = 1,
    ) -> None:
        
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            
        if generation_kwargs is None:
            generation_kwargs = {
                "temperature": 0.9,
                "do_sample": True,
                "max_new_tokens": 2048,
                "stop_sequences": [
                    "<|eot_id|>",
                    "<|start_header_id|>",
                    "assistant",
                    " \n\n",
                ],
            }
            
        if llm is None:
            llm = InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                magpie_pre_query_template="llama3",
                generation_kwargs=generation_kwargs,
                api_key=hf_token,
            )
            
        with Pipeline(name="sft") as pipeline:
            magpie = MagpieGenerator(
                llm=llm,
                n_turns=1,
                num_rows=10,
                batch_size=1,
                system_prompt=SYSTEM_PROMPT,
                output_mappings={"instruction": "prompt", "response": "completion"},
            )
            keep_columns = KeepColumns(
                columns=["prompt", "completion", "model_name"],
            )
            magpie.connect(keep_columns)

        self.pipeline = pipeline

    def run(self):
        return self.pipeline.run()
