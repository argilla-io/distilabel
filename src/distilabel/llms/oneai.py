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
from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import Field, PrivateAttr, SecretStr, validate_call

from distilabel.llms.base import AsyncLLM
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import FormattedInput, InstructorStructuredOutputType
  
if TYPE_CHECKING:
    from openai import AsyncOpenAI

_ONEAI_API_KEY_ENV_VAR_NAME = "01AI_API_KEY"
  
  
class OneAI(AsyncLLM):  
    """The 01.AI API platform enables developers to integrate advanced natural language processing capabilities into their own applications. Developers can utilize the AI capabilities of the Yi series LLMs to perform a variety of tasks, such as text generation, language translation, content summarization, logical reasoning, mathematical calculation, and code generation.

    The 01.AI API platform provides flexible calling methods, supports various programming languages, and can be customized with features to meet the needs of different scenarios. Both individual and enterprise developers can unlock new approaches to innovate, improve user experience and drive business growth.

    In addition, the 01.AI API platform also provides detailed documentation and sample code to help developers quickly get started and utilize the capabilities of the Yi series LLMs effectively.

    Attributes:
        model: the model name to use for the LLM, e.g., `yi-large`.
        base_url: the base URL to use for the OneAI API requests. Defaults to `None`, which
            means that the value set for the environment variable `01AI_BASE_URL` will be used, or
            "https://api.01.ai/v1" if not set.
        api_key: the API key to authenticate the requests to the OneAI API. Defaults to `None` which
            means that the value set for the environment variable `01AI_API_KEY` will be used, or
            `None` if not set.
        max_retries: the maximum number of times to retry the request to the API before failing.
        timeout: the maximum time in seconds to wait for a response from the API.  
        structured_output: the structured output format to use across all the generations.  
        _api_key_env_var: the name of the environment variable to use for the API key.
            It is meant to be used internally.  

    Examples:  

        Generate Json Outputs you can use in "function call" pipelines:  
  
        ```python
        from distilabel.steps.tasks import TextGeneration
        from distilabel.llms.oneai import OneAI

        text_gen = TextGeneration(
            llm = OneAI(api_key=os.getenv("01AI_BASE_URL", None))  # yi-large is the default model
        )

        text_gen.load()

        wordphrases = "During his presidency, a number of improvements to the campus were made. The Georgetown University Hospital was opened and the first patient was accepted."

        metadata_prompt = f"WORD PHRASES:\n\n{wordphrases}\n\n you will recieve a text or a question, produce metadata operator pairs for the text . ONLY PROVIDE THE FINAL JSON , DO NOT PRODUCE ANY ADDITION INSTRUCTION , ONLY PRODUCE ONE METADATA STRING PER OPERATOR:"

        result = next(
            text_gen.process(
                [{"instruction": metadata_prompt}]
            )
        )
        # result
        # [
        #     {
        #         'instruction': 'your instruction',
        #         'model_name': 'yi-large',
        #         'generation': 'generation',
        #     }
        # ]
        ```
    """

    model: Optional[str]  = "yi-large"
    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "01AI_BASE_URL", "https://api.lingyiwanwu.com/v1"
        ),
        description="The base URL to use for the OneAI API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_ONEAI_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the OneAI API.",
    )
    max_retries: RuntimeParameter[int] = Field(
        default=6,
        description="The maximum number of times to retry the request to the API before failing.",
    )
    timeout: RuntimeParameter[int] = Field(
        default=120,
        description="The maximum time in seconds to wait for a response from the API.",
    )
    structured_output: Optional[RuntimeParameter[InstructorStructuredOutputType]] = Field(  
        default=None,  
        description="The structured output format to use across all the generations.",
    )

    _api_key_env_var: str = PrivateAttr(_ONEAI_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["AsyncOpenAI"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `AsyncOpenAI` client to benefit from async requests."""
        super().load()
        try:
            from openai import AsyncOpenAI
        except ImportError as ie:
            raise ImportError(
                "01ai in Distillabel relies on the OpenAI Python client which is not installed. Please install it using `pip install openai`."
            ) from ie

        if self.api_key is None:
            raise ValueError(
                f"To use `{self.__class__.__name__}` an API key must be provided via `api_key`"
                f" attribute or runtime parameter, or set the environment variable `{self._api_key_env_var}`."
            )
 
        self._aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        if self.structured_output:
            result = self._prepare_structured_output(
                structured_output=self.structured_output,
                client=self._aclient,
                framework="openai",
            )
            self._aclient = result.get("client")
            if structured_output := result.get("structured_output"):
                self.structured_output = structured_output

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    @validate_call
    async def agenerate(
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[str] = None,
    ) -> GenerateOutput:
        """Generates text using the OneAI LLM."""

        structured_output = None
        if isinstance(input, tuple):
            input, structured_output = input
            result = self._prepare_structured_output(
                structured_output=structured_output,
                client=self._aclient,
                framework="openai",
            )
            self._aclient = result.get("client")
        if structured_output is None and self.structured_output is not None:
            structured_output = self.structured_output

        kwargs = {
            "messages": input,
            "model": self.model,
            "max_tokens": max_new_tokens,
            "n": num_generations,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "timeout": 50,
        }

        if response_format is not None:
            if response_format not in ["text", "json", "json_object"]:
                raise ValueError(
                    f"Invalid response format '{response_format}'. Must be either 'text' or 'json'."
                )
            if response_format == "json":
                response_format = "json_object"
            kwargs["response_format"] = response_format

        if structured_output:
            kwargs = self._prepare_kwargs(kwargs, structured_output)

        generations = []
        completion = await self._aclient.chat.completions.create(**kwargs)
        if structured_output:
            generations.append(completion.model_dump_json())
            return generations

        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(
                    f"Received no response using the 01ai client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
        return generations
