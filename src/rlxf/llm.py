from typing import Union
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
        self.model = model.to(self.device)
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


class Llama2CppLLM:
    """
    Examples:
        >>> from llama_cpp import Llama
        >>> llm = Llama2CppLLM(model_file="./llama-2-7b-chat.Q4_0.gguf")
        >>> llm.batch_generate(["What is the name of the capital of France?"])
    """
    def __init__(self, model_file: str) -> None:
        self.model = Llama(model_path=model_file, n_gpu_layers=1)

    def batch_generate(self, texts: list[str]) -> list[str]:
        return [self.model.create_completion(Llama2Prompt.chat_format(text), max_tokens=32, temperature=0.0) for text in texts]
