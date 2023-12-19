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

import torch
from datasets import Dataset
from distilabel.llm import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks.critique.prometheus import PrometheusTask
from transformers import AutoTokenizer, LlamaForCausalLM

if __name__ == "__main__":
    model = LlamaForCausalLM.from_pretrained(
        "kaist-ai/Prometheus-7b-v1.0", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token=os.getenv("HF_TOKEN")
    )
    pipeline = Pipeline(
        labeller=TransformersLLM(
            model=model,  # type: ignore
            tokenizer=tokenizer,
            task=PrometheusTask(
                scoring_criteria="Is the provided completion accurate based on the given instruction?",
                score_descriptions={
                    0: "Totaly off-topic and inaccurate",
                    1: "Incorrect and inaccurate",
                    2: "Almost correct, but partially inaccurate",
                    3: "Correct but badly phrased",
                    4: "Correct and accurate",
                },
            ),
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=512,
        ),
    )

    dataset = Dataset.from_dict(
        {
            "instruction": ["What's the capital of Spain?"],
            "completion": ["Paris"],
            "ref_completion": ["Madrid"],
        }
    )

    dataset = pipeline.generate(
        dataset,  # type: ignore
        display_progress_bar=True,
        skip_dry_run=True,
    )
