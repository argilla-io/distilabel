import json
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union
from urllib import request

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class OllamaLLM(LLM):
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def __init__(
        self,
        model: str,
        task: "Task",
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the OllamaLLM class and align with https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values"""
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        self.max_tokens = max_new_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

        self.model = model

        self._api_available()
        self._api_model_available()

    def _api_available(self):
        """calls GET {OLLAMA_HOST}"""
        response = request.urlopen(self.OLLAMA_HOST)
        if response.getcode() != 200:
            raise ValueError(
                f"Could not connect to Ollama as {self.OLLAMA_HOST}. Check https://github.com/jmorganca/ollama for deployment guide."
            )

    def _api_model_available(self):
        if (
            self._api_generate(prompt=[{"role": "user", "content": "hi"}], max_tokens=1)
            is None
        ):
            raise ValueError(
                f"Model {self.model} is not available. Run `ollama run {self.clean_model}` to serve the model."
            )

    def _api_generate(self, prompt: str, n: int = 0, **kwargs) -> str:
        """Calls POST {OLLAMA_HOST}/api/chat"""
        # Request payload
        payload = {
            "model": self.model,
            "messages": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens") or self.max_tokens,
                "repeat_penalty": self.frequency_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

        # Convert payload to JSON
        data = json.dumps(payload).encode("utf-8")

        # Create the request
        url = f"{self.OLLAMA_HOST}/api/chat"
        req = request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        try:
            # Send the request
            with request.urlopen(req) as response:
                # Check if the request was successful (status code 200)
                if response.getcode() == 200:
                    # Parse and return the response JSON
                    return json.loads(response.read().decode("utf-8"))
                elif response.getcode() >= 500 and n < 5:
                    # If the request failed, try again
                    time.sleep(1)
                    return self._api_generate(prompt, n + 1)
                else:
                    raise ValueError(
                        f"Ollama API request failed with status_code {response.getcode()}."
                    )
        except Exception as e:
            raise ValueError("Error in Ollama API request") from e

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "num_ctx": self.max_tokens,
                "frequency_penalty": self.frequency_penalty,
                "repeat_penalty": self.presence_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )

    @property
    def model_name(self) -> str:
        """Returns the name of the Ollama model."""
        return self.model

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List[LLMOutput]]:
        prompts = self._generate_prompts(inputs, default_format="openai")
        outputs = []
        for prompt in prompts:
            responses = [
                self._api_generate(prompt=prompt) for _ in range(num_generations)
            ]

            output = []
            for response in responses:
                raw_output = response.get("message", {}).get("content")
                try:
                    parsed_response = self.task.parse_output(raw_output.strip())
                except Exception as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=raw_output,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
