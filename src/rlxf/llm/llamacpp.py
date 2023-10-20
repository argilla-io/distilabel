from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generator, Optional

from rlxf.llm.base import LLM
from rlxf.prompts.llama import Llama2Prompt

if TYPE_CHECKING:
    from llama_cpp import Llama


class LlamaCppLLM(LLM):
    """
    Examples:
        >>> from llama_cpp import Llama
        >>> model = Llama(model_path="./llama-2-7b-chat.Q4_0.gguf", n_gpu_layers=1)
        >>> llm = LlamaCppLLM(model=model)
        >>> llm.batch_generate(["What is the name of the capital of France?"])
    """

    def __init__(
        self,
        model: Llama,
        prompt_formatting_fn: Optional[Callable] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> None:
        self.model = model
        self.prompt_formatting_fn = prompt_formatting_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_return_sequences = num_return_sequences

    def batch_generate(
        self, prompts: list[str], responses: list[list[str]] | None = None
    ) -> Generator[list[str], None, None]:
        """
        Note:
            The completion in `llama-cpp-python` may eventually contain the input prompt,
            but it does not remove that consistently, so we may need to develop something
            on top to fix it.
        """
        for prompt, responses_ in self._batch_iterator(prompts, responses):
            sequences = []
            for _ in range(self.num_return_sequences):
                if self.prompt_formatting_fn is not None:
                    text = self.prompt_formatting_fn(prompt, responses_)
                else:
                    text = prompt
                generation = self.model.create_completion(
                    text, max_tokens=self.max_new_tokens, temperature=self.temperature
                )["choices"][0]["text"].strip()
                sequences.append(generation)
            yield sequences

    @classmethod
    def as_generator(cls, model: Llama, **kwargs: Any) -> "LlamaCppLLM":
        """Classmethod with some helper defaults to act as a response generator for any
        given prompt.
        """
        return cls(model=model, prompt_formatting_fn=Llama2Prompt.chat_format, **kwargs)

    @classmethod
    def as_ranker(cls, model: Llama, **kwargs: Any) -> "LlamaCppLLM":
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
        return cls(model=model, prompt_formatting_fn=Llama2Prompt.rank_format, **kwargs)
