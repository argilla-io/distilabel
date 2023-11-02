import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import replicate
from replicate.exceptions import ReplicateError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ultralabel.llm.base import LLM

if TYPE_CHECKING:
    from ultralabel.tasks.base import Task


_REPLICATE_API_RETRY_ON_EXCEPTIONS = ReplicateError
_REPLICATE_API_STOP_AFTER_ATTEMPT = 6
_REPLICATE_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_REPLICATE_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10


class ReplicateLLM(LLM):
    def __init__(
        self,
        endpoint_url: str,
        task: "Task",
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        num_threads: Union[int, None] = None,
        formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_threads=num_threads,
            formatting_fn=formatting_fn,
        )
        self.token = os.environ.get("REPLICATE_API_TOKEN")
        assert (
            self.token is not None
        ), "The `REPLICATE_API_TOKEN` environment variable must be set to use Replicate."

        self.endpoint_url = endpoint_url

    @retry(
        retry=retry_if_exception_type(_REPLICATE_API_RETRY_ON_EXCEPTIONS),
        stop=stop_after_attempt(_REPLICATE_API_STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(
            multiplier=_REPLICATE_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER,
            max=_REPLICATE_API_WAIT_RANDOM_EXPONENTIAL_MAX,
        ),
    )
    def _text_generation_with_backoff(self, prompt: str) -> Any:
        return replicate.run(
            self.endpoint_url,
            input={
                "prompt": prompt,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
        )

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        prompt = self.task.generate_prompt(**input)
        if self.formatting_fn is not None:
            prompt = self.formatting_fn(prompt)
        raw_responses = [
            "".join(list(self._text_generation_with_backoff(prompt=prompt)))
            for _ in range(num_generations)
        ]
        parsed_responses = []
        for response in raw_responses:
            try:
                parsed_response = self.task.parse_output(response)
            except Exception as e:
                warnings.warn(
                    f"Error parsing Replicate output: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                parsed_response = {}
            parsed_responses.append(parsed_response)
        return raw_responses, parsed_responses
