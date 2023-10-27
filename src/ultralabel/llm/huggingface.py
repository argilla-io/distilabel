import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
from huggingface_hub import InferenceClient, InferenceTimeoutError
from huggingface_hub.inference._text_generation import TextGenerationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from ultralabel.llm.base import LLM

if TYPE_CHECKING:
    from ultralabel.prompts.base import PromptTemplate


_INFERENCE_ENDPOINTS_API_RETRY_ON_EXCEPTIONS = (
    InferenceTimeoutError,
    TextGenerationError,
)
_INFERENCE_ENDPOINTS_API_STOP_AFTER_ATTEMPT = 6
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10


class TransformersLLM(LLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_template: "PromptTemplate",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_threads: Union[int, None] = None,
    ) -> None:
        super().__init__(
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_threads=num_threads,
        )

        self.model = model
        if self.device != "cpu":
            self.model.to(self.device)

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def _generate(self, input: Dict[str, Any], num_generations: int = 1) -> Any:
        prompt = self.prompt_template.generate_prompt(**input)
        encoding = self.tokenizer(text=prompt, padding=True, return_tensors="pt")
        if self.device != "cpu":
            encoding = encoding.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **encoding,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    num_generations=num_generations,
                ),
            )
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids[:, -(encoding.input_ids.shape[1]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [
            self.prompt_template.parse_output(decoded_output)
            for decoded_output in decoded_outputs
        ]


class InferenceEndpointsLLM(LLM):
    def __init__(
        self,
        endpoint_url: str,
        prompt_template: "PromptTemplate",
        token: Union[str, None] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        num_threads: Union[int, None] = None,
    ) -> None:
        super().__init__(
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_threads=num_threads,
        )

        token = token or os.environ.get("HF_TOKEN")
        assert (
           token is not None
        ), "Either the `token` arg or the `HF_TOKEN` environment variable must be set to use HF Inference Endpoints."


        self.client = InferenceClient(model=endpoint_url, token=token)

    @retry(
        retry=retry_if_exception_type(_INFERENCE_ENDPOINTS_API_RETRY_ON_EXCEPTIONS),
        stop=stop_after_attempt(_INFERENCE_ENDPOINTS_API_STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(
            multiplier=_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER,
            max=_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MAX,
        ),
    )
    def _text_generation_with_backoff(self, **kwargs: Any) -> Any:
        return self.client.text_generation(
            **kwargs, max_new_tokens=self.max_new_tokens, temperature=self.temperature
        )

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> List[Dict[str, Any]]:
        prompt = self.prompt_template.generate_prompt(**input)
        responses = [
            self._text_generation_with_backoff(prompt=prompt)
            for _ in range(num_generations)
        ]
        return [self.prompt_template.parse_output(response) for response in responses]
