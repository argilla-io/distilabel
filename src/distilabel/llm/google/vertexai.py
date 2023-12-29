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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from distilabel.llm import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _VERTEXAI_AVAILABLE

if _VERTEXAI_AVAILABLE:
    from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

if TYPE_CHECKING:
    from vertexai.preview.generative_models import GenerationResponse

    from distilabel.tasks.base import Task

logger = get_logger()


class VertexGenerativeModelLLM(LLM):
    """An `LLM` which allows to use models from the Gemini API of Vertex AI."""

    def __init__(
        self,
        task: "Task",
        model: str = "gemini-pro",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: int = 128,
        stop_sequences: Optional[List[str]] = None,
        num_threads: Union[int, None] = None,
    ) -> None:
        """Initializes the `VertexGenerativeModelLLM` class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "gemini-pro".
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to 1.0.
            top_k (int, optional): the top-k value to be used for generation.
                Defaults to 40.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
                Defaults to `None`.
        """
        super().__init__(task=task, num_threads=num_threads)

        if not _VERTEXAI_AVAILABLE:
            raise ImportError(
                "`VertexAILLM` cannot be used as `google-cloud-aiplatform` is not installed,"
                " please install it with `pip install google-cloud-aiplatform`"
            )

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_new_tokens
        self.stop_sequences = stop_sequences

        # TODO: check if there's any endpoint to get available models
        if model not in ["gemini-pro", "gemini-pro-vision"]:
            raise ValueError(
                f"Model '{model}' is not available in the Gemini API of Vertex AI."
            )
        self.model = GenerativeModel(model)

    @property
    def model_name(self) -> str:
        """Returns the name of the model used for generation."""
        return self.model._model_name

    def _generate_contents(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generates a list of valid dicts that can be parsed to `vertexai.preview.generative_models.Content`
        objects for each input.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.

        Returns:
            List[List[Dict[str, Any]]]: the list of valid `vertexai.preview.generative_models.Content`
                objects.
        """
        contents = []
        for input in inputs:
            prompt = self.task.generate_prompt(**input)
            contents.append(
                [
                    {"role": "user", "parts": [{"text": prompt.system_prompt}]},
                    {"role": "model", "parts": [{"text": "Understood."}]},
                    {"role": "user", "parts": [{"text": prompt.formatted_prompt}]},
                ]
            )
        return contents

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        inputs_contents = self._generate_contents(inputs)
        outputs = []
        for contents in inputs_contents:
            output = []
            # TODO: remove this for-loop once `GenerationConfig.candidate_count` valid range is not [1, 2)
            for _ in range(num_generations):
                response: "GenerationResponse" = self.model.generate_content(  # type: ignore
                    contents=contents,
                    generation_config=GenerationConfig(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        # TODO: update this parameter to have `num_generations` as value once `GenerationConfig.candidate_count` valid range is not [1, 2)
                        candidate_count=1,
                        max_output_tokens=self.max_output_tokens,
                        stop_sequences=self.stop_sequences,
                    ),
                )

                try:
                    parsed_response = self.task.parse_output(response.text)
                except Exception as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    parsed_response = None

                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=contents,
                        raw_output=response.text,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
