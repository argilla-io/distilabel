from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from vllm import SamplingParams

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger

if TYPE_CHECKING:
    from vllm import LLM as _vLLM

    from distilabel.tasks.base import Task

logger = get_logger()


class vLLM(LLM):
    def __init__(
        self,
        vllm: "_vLLM",
        task: "Task",
        formatting_fn: Union[Callable[..., str], None] = None,
        max_new_tokens: int = 128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> None:
        super().__init__(task=task, formatting_fn=formatting_fn)

        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_new_tokens

        self.vllm = vllm

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List[LLMOutput]]:
        prompts = self._generate_prompts(inputs)
        requests = self.vllm.generate(
            prompts,
            SamplingParams(
                n=num_generations,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
            ),
            use_tqdm=False,
        )
        outputs = []
        for request, prompt in zip(requests, prompts):
            output = []
            for request_output in request.outputs:
                try:
                    parsed_output = self.task.parse_output(request_output.text)
                except Exception as e:
                    logger.error(f"Error parsing vLLM output: {e}")
                    parsed_output = None
                output.append(
                    LLMOutput(
                        prompt_used=prompt,
                        raw_output=request_output.text,
                        parsed_output=parsed_output,
                    )
                )
            outputs.append(output)
        return outputs
