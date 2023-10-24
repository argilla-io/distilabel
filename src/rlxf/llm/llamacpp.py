from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from rlxf.llm.base import LLM

if TYPE_CHECKING:
    from llama_cpp import Llama

    from rlxf.prompts.base import PromptTemplate


class LlamaCppLLM(LLM):
    def __init__(
        self,
        model: Llama,
        prompt_template: "PromptTemplate",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(prompt_template)

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[Dict[str, Any]]:
        generations = []
        for input in inputs:
            input = self.prompt_template.generate_prompt(**input)
            input_generations = []
            for _ in range(num_generations):
                output = self.model.create_completion(
                    input, max_tokens=self.max_new_tokens, temperature=self.temperature
                )["choices"][0]["text"].strip()
                output = self.prompt_template.parse_output(output)
                input_generations.append(output)
            generations.append(input_generations)
        return generations

    @property
    def return_futures(self) -> bool:
        return False
