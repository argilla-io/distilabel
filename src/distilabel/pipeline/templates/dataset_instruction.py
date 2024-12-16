# TODO: This license is not consistent with the license used in the project.
#       Delete the inconsistent license and above line and rerun pre-commit to insert a good license.
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
from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import ExpandColumns, KeepColumns
from distilabel.steps.tasks import SelfInstruct, TextGeneration

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


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
        """Initializes the pipeline.

        Args:
            llm (Optional[LLM], optional): The language model to use. Defaults to None.
            system_prompt (str, optional): The system prompt to use. Defaults to "You are a creative AI Assistant writer.".
            hf_token (Optional[str], optional): The Hugging Face API token to use. Defaults to None.
            num_instructions (int, optional): The number of instructions to generate. Defaults to 2.
            batch_size (int, optional): The batch size to use. Defaults to 1.
        """
        if llm is None:
            self.llm: LLM = InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                generation_kwargs={
                    "temperature": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 2048,
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
        """Runs the pipeline and returns a Distiset.

        Args:
            dataset: The dataset to run the pipeline on.
            **kwargs: Additional arguments to pass to the pipeline.
        """
        return self.pipeline.run(dataset, **kwargs)

    def _get_pipeline(
        self, system_prompt: str, num_instructions: int, batch_size: int
    ) -> Pipeline:
        """Returns a pipeline that generates instructions and responses for a given system prompt."""
        with Pipeline(name="dataset_chat") as pipeline:
            self_instruct = SelfInstruct(
                llm=self.llm,
                num_instructions=num_instructions,
            )

            expand_columns = ExpandColumns(
                columns=["instructions"],
                output_mappings={"instructions": "instruction"},
            )

            keep_instruction = KeepColumns(
                columns=["instruction", "input"],
            )

            response_generation = TextGeneration(
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
