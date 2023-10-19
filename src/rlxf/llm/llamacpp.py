from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator, Optional

from rlxf.llm.base import LLM
from rlxf.prompts.llama2 import Llama2Prompt

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
        self, model: Llama, prompt_formatting_fn: Optional[Callable] = None
    ) -> None:
        self.model = model
        self.prompt_formatting_fn = prompt_formatting_fn

    def batch_generate(
        self, prompts: list[str], responses: list[list[str]] | None = None
    ) -> Generator[str, None, None]:
        """
        Note:
            The completion in `llama-cpp-python` may eventually contain the input prompt,
            but it does not remove that consistently, so we may need to develop something
            on top to fix it.
        """
        if self.prompt_formatting_fn is not None:
            for prompt, responses_ in self._batch_iterator(prompts, responses):
                text = self.prompt_formatting_fn(prompt, responses_)
                yield self.model.create_completion(
                    text, max_tokens=50, temperature=0.0, echo=False
                )["choices"][0]["text"]

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
