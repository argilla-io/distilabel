import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _LITELLM_AVAILABLE

if _LITELLM_AVAILABLE:
    import requests

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class OllamaLLM(LLM):
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

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

        if not _LITELLM_AVAILABLE:
            raise ImportError(
                "`litellm` is not availble. Please install it using `pip install litellm` or `pip install 'distilabel[litellm]'."
            )

        self.max_tokens = max_new_tokens
        self.frequency_penalty = frequency_penalty
        # self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

        # if "ollama/" not in model and "ollama_chat/" not in model:
        #     raise ValueError(
        #         "ollama models must be prefixed with 'ollama/' or 'ollama_chat' for chat models."
        #     )
        # elif "ollama/" in model:
        #     self.chat_model = False
        # elif "ollama_chat/" in model:
        #     self.chat_model = True
        self.model = model

        # self.clean_model: str = model.replace("ollama/", "").replace("ollama_chat/", "")

    def _ollam_api_generate(self, prompt: str, n: int = 0, **kwargs) -> str:
        """Calls POST {OLLAMA_BASE_URL}/api/chat"""
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

        # Send the request
        response = requests.post(f"{self.OLLAMA_BASE_URL}/api/chat", json=payload)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse and return the response JSON
            return response.json()
        elif response.status_code >= 50 and n < 5:
            # If the request failed, try again
            return self._ollam_api_generate(prompt, n + 1)
        else:
            raise ValueError(
                f"Ollama API request failed with status_code {response.status_code}."
            )

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

    def _is_running(self):
        """calls GET {OLLAMA_BASE_URL}"""
        response = requests.get(self.OLLAMA_BASE_URL)
        if response.status_code != 200:
            raise ValueError(
                "Could not connect to Ollama. Check https://github.com/jmorganca/ollama for usage guide."
            )

    def _is_model_available(self):
        if (
            self._ollam_api_generate(
                prompt=[{"role": "user", "content": "hi"}], max_tokens=1
            )
            is None
        ):
            raise ValueError(
                f"Model {self.model} is not available. Run `ollama run {self.clean_model}` to serve the model."
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
                self._ollam_api_generate(prompt=prompt) for _ in range(num_generations)
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
