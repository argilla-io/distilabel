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

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from pydantic import Field, PrivateAttr, validate_call

from distilabel.llms.base import AsyncLLM
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from litellm import Choices


class LiteLLM(AsyncLLM):
    """LiteLLM implementation running the async API client.

    Attributes:
        model: the model name to use for the LLM e.g. "gpt-3.5-turbo" or "mistral/mistral-large",
            etc.
        verbose: whether to log the LiteLLM client's logs. Defaults to `False`.

    Runtime parameters:
        - `verbose`: whether to log the LiteLLM client's logs. Defaults to `False`.
    """

    model: str
    verbose: RuntimeParameter[bool] = Field(
        default=False, description="Whether to log the LiteLLM client's logs."
    )

    _aclient: Optional[Callable] = PrivateAttr(...)

    def load(self) -> None:
        """
        Loads the `acompletion` LiteLLM client to benefit from async requests.
        """
        super().load()

        try:
            import litellm

            litellm.telemetry = False
        except ImportError as e:
            raise ImportError(
                "LiteLLM Python client is not installed. Please install it using"
                " `pip install litellm`."
            ) from e
        self._aclient = litellm.acompletion

        if not self.verbose:
            litellm.suppress_debug_info = True
            for key in logging.Logger.manager.loggerDict.keys():
                if "litellm" not in key.lower():
                    continue
                logging.getLogger(key).setLevel(logging.CRITICAL)

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: ChatType,
        num_generations: int = 1,
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        stop: Optional[Union[str, list]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        metadata: Optional[dict] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,
        mock_response: Optional[str] = None,
        force_timeout: Optional[int] = 600,
        custom_llm_provider: Optional[str] = None,
    ) -> GenerateOutput:
        """Generates `num_generations` responses for the given input using the [LiteLLM async client](https://github.com/BerriAI/litellm).

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            functions: a list of functions to apply to the conversation messages. Defaults to
                `None`.
            function_call: the name of the function to call within the conversation. Defaults
                to `None`.
            temperature: the temperature to use for the generation. Defaults to `1.0`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            stop: Up to 4 sequences where the LLM API will stop generating further tokens.
                Defaults to `None`.
            max_tokens: The maximum number of tokens in the generated completion. Defaults to
                `None`.
            presence_penalty: It is used to penalize new tokens based on their existence in the
                text so far. Defaults to `None`.
            frequency_penalty: It is used to penalize new tokens based on their frequency in the
                text so far. Defaults to `None`.
            logit_bias: Used to modify the probability of specific tokens appearing in the
                completion. Defaults to `None`.
            user: A unique identifier representing your end-user. This can help the LLM provider
                to monitor and detect abuse. Defaults to `None`.
            metadata: Pass in additional metadata to tag your completion calls - eg. prompt
                version, details, etc. Defaults to `None`.
            api_base: Base URL for the API. Defaults to `None`.
            api_version: API version. Defaults to `None`.
            api_key: API key. Defaults to `None`.
            model_list: List of api base, version, keys. Defaults to `None`.
            mock_response: If provided, return a mock completion response for testing or debugging
                purposes. Defaults to `None`.
            force_timeout: The maximum execution time in seconds for the completion request.
                Defaults to `600`.
            custom_llm_provider: Used for Non-OpenAI LLMs, Example usage for bedrock, set(iterable)
                model="amazon.titan-tg1-large" and custom_llm_provider="bedrock". Defaults to
                `None`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        import litellm

        async def _call_aclient_until_n_choices() -> List["Choices"]:
            choices = []
            while len(choices) < num_generations:
                completion = await self._aclient(  # type: ignore
                    model=self.model,
                    messages=input,
                    n=num_generations,
                    functions=functions,
                    function_call=function_call,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                    stop=stop,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    user=user,
                    metadata=metadata,
                    api_base=api_base,
                    api_version=api_version,
                    api_key=api_key,
                    model_list=model_list,
                    mock_response=mock_response,
                    force_timeout=force_timeout,
                    custom_llm_provider=custom_llm_provider,
                )
                choices.extend(completion.choices)
            return choices

        # litellm.drop_params is used to en/disable sending **kwargs parameters to the API if they cannot be used
        try:
            litellm.drop_params = False
            choices = await _call_aclient_until_n_choices()
        except litellm.exceptions.APIError as e:
            if "does not support parameters" in str(e):
                litellm.drop_params = True
                choices = await _call_aclient_until_n_choices()
            else:
                raise e

        generations = []
        for choice in choices:
            if (content := choice.message.content) is None:
                self._logger.warning(
                    f"Received no response using LiteLLM client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
        return generations
