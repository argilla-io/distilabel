from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
from huggingface_hub import InferenceClient
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlxf.llm.base import LLM

if TYPE_CHECKING:
    from rlxf.prompts.base import PromptTemplate


class TransformersLLM(LLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_template: "PromptTemplate",
    ) -> None:
        super().__init__(prompt_template=prompt_template)

        self.model = model
        if self.device != "cpu":
            self.model.to(self.device)

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generate_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "num_return_sequences": 1,
        }

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def generate(self, inputs: List[Dict[str, Any]]) -> Any:
        prompts = [self.prompt_template.generate_prompt(**input) for input in inputs]
        batch_encoding = self.tokenizer(text=prompts, padding=True, return_tensors="pt")
        if self.device != "cpu":
            batch_encoding = batch_encoding.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **batch_encoding, **self.generate_kwargs
            )
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids[:, -(batch_encoding.input_ids.shape[1]) :],
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
    ) -> None:
        super().__init__(prompt_template=prompt_template)

        self.client = InferenceClient(model=endpoint_url, token=token)

    def generate(self, inputs: List[Dict[str, Any]]) -> Any:
        # TODO(alvarobartt): add `generation_kwargs`
        generations = []
        for input in inputs:
            prompt = self.prompt_template.generate_prompt(**input)
            generation = self.client.text_generation(prompt).generated_text
            generations.append(generation)
        return generations
