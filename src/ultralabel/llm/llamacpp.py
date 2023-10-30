from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ultralabel.llm.base import LLM

if TYPE_CHECKING:
    from llama_cpp import Llama

    from ultralabel.tasks.base import Task


class LlamaCppLLM(LLM):
    def __init__(
        self,
        model: Llama,
        task: "Task",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        super().__init__(
            task=task,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        self.model = model

    def _generate(self, input: Dict[str, Any], num_generations: int = 1) -> List[Any]:
        prompt = self.task.generate_prompt(**input)
        generations = []
        for _ in range(num_generations):
            output = self.model.create_completion(
                prompt, max_tokens=self.max_new_tokens, temperature=self.temperature
            )["choices"][0]["text"].strip()
            generations.append(self.task.parse_output(output))
        return generations
