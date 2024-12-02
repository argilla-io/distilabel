from typing import List
from pathlib import Path

from pydantic import BaseModel, Field

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.clustering import text_clustering
from distilabel.steps.tasks import TextGeneration, ChatGeneration

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

from typing import Optional

from distilabel.distiset import Distiset
from distilabel.llms import LLM, InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MagpieGenerator
from distilabel.steps import ExpandColumns, KeepColumns

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
from distilabel.steps.tasks import SelfInstruct
from distilabel.models import InferenceEndpointsLLM


class DatasetInstructionResponsePipeline:
    """Generates instructions and responses for a dataset with input documents.

    This example pipeline can be used for a Supervised Fine-Tuning dataset which you
    could use to train or evaluate a model. The pipeline generates instructions using the
    SelfInstruct step and TextGeneration step.

    References:
        - [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)

    Example:

        Generate instructions and responses for a given system prompt:

        ```python
        from datasets import Dataset
        from distilabel.pipeline import DatasetInstructionResponsePipeline

        pipeline = DatasetInstructionResponsePipeline(num_instructions=5)

        distiset = pipeline.pipeline.run(
            use_cache=False,
            dataset=Dataset.from_list(
                mapping=[
                    {
                        "input": "<document>",
                    }
                ]
            ),
        )
        ```
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        system_prompt: str = "You are a creative AI Assistant writer.",
        hf_token: Optional[str] = None,
        num_instructions: int = 2,
        batch_size: int = 1,
    ) -> None:
        if llm is None:
            self.llm: LLM = InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 2048,
                    "stop_sequences": [
                        "<|eot_id|>",
                        "<|start_header_id|>",
                        "assistant",
                        " \n\n",
                    ],
                },
                api_key=hf_token,
            )
        else:
            self.llm = llm

        self.pipeline: Pipeline = self._get_pipeline(
            system_prompt=system_prompt,
            num_instructions=num_instructions,
            batch_size=batch_size,
        )

    def run(self, dataset, **kwargs) -> Distiset:
        """Runs the pipeline and returns a Distiset."""
        return self.pipeline.run(dataset, **kwargs)

    def _get_pipeline(
        self, system_prompt: str, num_instructions: int, batch_size: int
    ) -> Pipeline:
        """Returns a pipeline that generates instructions and responses for a given system prompt."""
        with Pipeline(name="dataset_chat") as pipeline:

            self_instruct = SelfInstruct(
                llm=self.llm,
                num_instructions=num_instructions,  # This is the default value
            )

            expand_columns = ExpandColumns(
                columns=["instructions"],
                output_mappings={"instructions": "instruction"},
            )

            keep_instruction = KeepColumns(
                columns=["instruction", "input"],
            )

            response_generation = TextGeneration(  #
                name="exam_generation",
                system_prompt=system_prompt,
                template="Respond to the instruction based on the document. Document:\n{{ input }} \nInstruction: {{ instruction }}",
                llm=self.llm,
                input_batch_size=batch_size,
                output_mappings={"generation": "response"},
            )

            keep_response = KeepColumns(
                columns=["input", "instruction", "response"],
            )

            (
                self_instruct
                >> expand_columns
                >> keep_instruction
                >> response_generation
                >> keep_response
            )

        return pipeline

    def _get_output_columns(self, n_turns: int) -> list:
        """Returns the output mappings for the pipeline."""
        if n_turns == 1:
            return ["instruction", "response", "model_name"]
        else:
            return ["instruction", "conversation", "model_name"]
