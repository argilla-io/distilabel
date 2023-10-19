from __future__ import annotations
from typing import Callable, Union, Optional, Generator, List
from functools import cached_property

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from llama_cpp import Llama
from rlxf.prompts.llama2 import Llama2Prompt


class HuggingFaceLLM:
    """
    Examples:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        >>> llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
        >>> llm.batch_generate(["What is the name of the capital of France?"])
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> None:
        self.model = model
        if self.device != "cpu":
            self.model.to(self.device)
        self.tokenizer = tokenizer

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "num_return_sequences": num_return_sequences,
            "num_beams": num_return_sequences or 1,
        }

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def batch_generate(self, texts: list[str]) -> list[str]:
        batch_encoding = self.tokenizer(text=texts, padding=True, return_tensors="pt")
        if self.device != "cpu":
            batch_encoding = batch_encoding.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**batch_encoding, **self.generate_kwargs)
        return self.tokenizer.batch_decode(
            generated_ids[:, -(batch_encoding.input_ids.shape[1]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


class LlamaCppLLM:
    """
    Examples:
        >>> from llama_cpp import Llama
        >>> model = Llama(model_path="./llama-2-7b-chat.Q4_0.gguf", n_gpu_layers=1)
        >>> llm = LlamaCppLLM(model=model)
        >>> llm.batch_generate(["What is the name of the capital of France?"])
    """
    def __init__(self, model: Llama, prompt_formatting_fn: Optional[Callable] = None) -> None:
        self.model = model
        self.prompt_formatting_fn = prompt_formatting_fn

    def batch_generate(self, prompts: List[str], responses: List[List[str]] | None = None) -> Generator[str, None, None]:
        """
        Note:
            The completion in `llama-cpp-python` may eventually contain the input prompt,
            but it does not remove that consistently, so we may need to develop something
            on top to fix it.
        """
        if self.prompt_formatting_fn is not None:
            for prompt, responses_ in zip(prompts, responses if responses is not None else range(len(prompts))):
                text = self.prompt_formatting_fn(prompt, responses_)
                yield self.model.create_completion(text, max_tokens=32, temperature=0.0, echo=False)["choices"][0]["text"]

    @classmethod
    def as_generator(cls, model: Llama) -> "LlamaCppLLM":
        """Classmethod with some helper defaults to act as a response generator for any
        given prompt.
        """
        return cls(
            model=model,
            prompt_formatting_fn=Llama2Prompt.chat_format,
        )
    
    @classmethod
    def as_ranker(cls, model: Llama) -> "LlamaCppLLM":
        """Classmethod with some helper defaults to act as a response ranker for any
        given collection of responses.

        Examples:
            >>> model = Llama(model_path="./llama-2-7b-chat.Q4_0.gguf", n_gpu_layers=1, verbose=False)
            >>> ranker = LlamaCppLLM.as_ranker(model=model)
            >>> output = ranker.batch_generate(prompts=["What is the capital city of Spain?"], responses=[["Madrid", "Barcelona", "Seville", "Valencia"]])
            >>> def parse_rank_output(output: str) -> List[str]:
            ...     return [["Madrid", "Barcelona", "Seville", "Valencia"][int(rank) - 1] for rank in output["choices"][0]["text"].split(">")]
            >>> print(parse_rank_output(output))
            ['Madrid', 'Barcelona', 'Seville', 'Valencia']
        """
        return cls(
            model=model,
            prompt_formatting_fn=Llama2Prompt.rank_format,
        )
