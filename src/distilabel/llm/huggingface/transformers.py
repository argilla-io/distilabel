import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from distilabel.llm.base import LLM
from distilabel.logger import get_logger

if TYPE_CHECKING:
    from distilabel.llm.utils import LLMOutput
    from distilabel.tasks.base import Task

logger = get_logger()


class TransformersLLM(LLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: "Task",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
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

        self.model = model
        if self.device != "cpu":
            self.model.to(self.device)

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if (
            hasattr(self.tokenizer, "use_default_system_prompt")
            and self.tokenizer.use_default_system_prompt
        ):
            # The `tokenizer` also has a method named `apply_chat_template` that expects a `Conversation` as OpenAI does with the ChatML format
            warnings.warn(
                "The provided `tokenizer` has `use_default_system_prompt=True` which means that the default system prompt will be used, which may collide with the `task` provided as an arg to this class.",
                UserWarning,
                stacklevel=2,
            )

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> List[LLMOutput]:
        prompt = self.task.generate_prompt(**input)
        if self.formatting_fn is not None:
            prompt = self.formatting_fn(prompt)
        encoding = self.tokenizer(text=prompt, padding=True, return_tensors="pt")
        if self.device != "cpu":
            encoding = encoding.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **encoding,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    num_generations=num_generations,
                ),
            )
        raw_outputs = self.tokenizer.batch_decode(
            generated_ids[:, -(encoding.input_ids.shape[1]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        outputs = []
        for raw_output in raw_outputs:
            try:
                parsed_output = self.task.parse_output(raw_output)
            except Exception as e:
                logger.error(f"Error parsing Transformers output: {e}")
                parsed_output = None
            outputs.append(
                LLMOutput(
                    prompt_used=prompt,
                    raw_output=raw_output,
                    parsed_output=parsed_output,
                )
            )
        return outputs
