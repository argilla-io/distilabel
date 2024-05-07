# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
)

from pydantic import BaseModel, Field

from distilabel.utils.serialization import _Serializable

if TYPE_CHECKING:
    from outlines.samplers import Sampler

    from distilabel.llms.azure import AzureOpenAILLM
    from distilabel.llms.huggingface.transformers import TransformersLLM
    from distilabel.llms.llamacpp import LlamaCppLLM
    from distilabel.llms.openai import OpenAILLM
    from distilabel.llms.typing import GenerateOutput
    from distilabel.llms.vllm import vLLM

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


OutputType = Literal["text", "json", "regex", "cfg"]
StructureType = Union[str, BaseModel, Callable]

# Restrict the LLMs that can be used with Outlines
AllowedLLMs = Union[
    "TransformersLLM", "LlamaCppLLM", "vLLM", "OpenAILLM", "AzureOpenAILLM"
]
SamplerType = Literal["greedy", "multinomial", "beam"]


class OutlinesStructuredOutput(BaseModel, _Serializable):
    """Integration of `outlines` library to generate structured outputs from LLMs.

    In general the user doesn't need to know about the `outlines` library, but it can
    be instantiated and passed directly to the `LLM`.

    Attributes:
        llm: the LLM model to use for the generation.
        sampler: the sampler to use for the generation. Defaults to `"multinomial"`.
        output_format: the format of the structured output. Defaults to `"text"`.
        output_structure: the structure of the output. Defaults to `None`.
        whitespace_pattern: the pattern to use to split the output into structured parts. Defaults to `None`.
        num_generations: the number of generations to produce. Defaults to `1`.
        top_k: the number of top tokens to sample from. Defaults to `None`.
        top_p: the cumulative probability threshold for sampling from the top tokens. Defaults to `None`.
        temperature: the temperature to use for sampling. Defaults to `None`.
    """

    llm: Any = Field(
        default=None,
        description="The LLM model from `outlines` to use for the generation.",
    )
    sampler: SamplerType = "multinomial"
    output_format: OutputType = "text"
    output_structure: Optional[StructureType] = None
    whitespace_pattern: Optional[str] = None
    num_generations: int = 1
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if type(self.output_structure) == type(BaseModel):
            # Force passing the schema as a string to simplify the transformation to a dataset afterwards.
            output_structure = self.output_structure.model_json_schema()
            output_structure.pop("required")
            output_structure = json.dumps(output_structure)
            # update the type in case it is a BaseModel to generate json.
            self.output_structure = output_structure

    def load(self):
        try:
            import outlines.generate as outlines_generate
        except ImportError as ie:
            raise ImportError(
                "`outlines` is not installed. Please install it using `pip install outlines` to use this class."
            ) from ie

        sampler_func = self._prepare_sampler(
            self.sampler,
            num_generations=self.num_generations,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
        )

        if self.output_format == "text":
            self._structured_generator = outlines_generate.text(
                self.llm, sampler=sampler_func
            )

        elif self.output_format == "json":
            if self.output_structure is None:
                # This should works like "json mode" in OpenAI.
                # https://outlines-dev.github.io/outlines/reference/json_mode/
                raise NotImplementedError(
                    "JSON Mode is not working currently on outlines."
                )
                import outlines

                self._structured_generator = outlines_generate.cfg(
                    self.llm, outlines.grammars.json, sampler=sampler_func
                )
            else:
                self._structured_generator = outlines_generate.json(
                    self.llm,
                    self.output_structure,  # schema object
                    sampler=sampler_func,
                    whitespace_pattern=self.whitespace_pattern,
                )

        elif self.output_format == "regex":
            if not self.output_structure:
                raise ValueError(
                    "`output_structure` must be provided for `regex` output_format."
                )
            self._structured_generator = outlines_generate.regex(
                self.llm,
                self.output_structure,  # regex string or re.compile?
                sampler=sampler_func,
            )
        elif self.output_format == "cfg":
            raise NotImplementedError(
                "There's a bug in `outlines` and we cannot work with `cfg` for the moment."
            )
            self._structured_generator = outlines_generate.cfg(
                self.llm, self.output_structure, sampler=sampler_func
            )
        else:
            raise NotImplementedError(
                f"Only {get_args(OutputType)} are supported for `outlines`."
            )

    def _prepare_sampler(
        self,
        sampler_name: SamplerType,
        num_generations: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> "Sampler":
        """Creates the sampler based on the name and the arguments. With default to `multinomial`.

        Args:
            sampler_name: One of "greedy", "multinomial" or "beam". Will default to "multinomial".
            num_generations: Defaults to 1.
            top_k: Same interpretation of `Transformers` used in the `multinomial` sampler. Defaults to None.
            top_p: Same interpretation of `Transformers` used in the `multinomial` sampler. Defaults to None.
            temperature: Same interpretation of `Transformers` used in the `multinomial` sampler. Defaults to None.

        Raises:
            ValueError: If he sampler_name is not one of the allowed samplers.

        Returns:
            Instance of a sampler to use for the generator.
        """
        # TODO: Place extra controls here to determine how to create the sampler allowed depending on the client.
        from outlines.samplers import beam_search, greedy, multinomial

        if sampler_name == "multinomial":
            return multinomial(
                samples=num_generations,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        elif sampler_name == "greedy":
            return greedy()
        elif sampler_name == "beam":
            return beam_search(beams=num_generations)

        raise ValueError(
            "Only `greedy`, `multinomial` and `beam` samplers are supported."
        )

    @classmethod
    def from_transformers(
        cls,
        llm: "TransformersLLM",
        sampler: SamplerType = "multinomial",
        output_format: OutputType = "text",
        output_structure: Optional[StructureType] = None,
        whitespace_pattern: Optional[str] = None,
        num_generations: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Self:
        """Creates an `OutlinesStructuredOutput` from a `TransformersLLM`.

        Args:
            llm: The `TransformersLLM` instance to use for the generation.
            sampler: the sampler for the logits. Defaults to "multinomial".
            output_format: Structured output wanted from the `LLM`. Defaults to "text".
            output_structure: the structure of the output. Defaults to `None`.
            whitespace_pattern: the pattern to use to split the output into structured parts. Defaults to `None`.
            num_generations: the number of generations to produce. Defaults to `1`.
            top_k: the number of top tokens to sample from. Defaults to `None`.
            top_p: the cumulative probability threshold for sampling from the top tokens. Defaults to `None`.
            temperature: the temperature to use for sampling. Defaults to `None`.

        Returns:
            `OutlinesStructuredOutput` instance.
        """
        from outlines.models.transformers import Transformers

        return cls(
            llm=Transformers(llm._pipeline.model, llm._pipeline.tokenizer),
            sampler=sampler,
            output_format=output_format,
            output_structure=output_structure,
            whitespace_pattern=whitespace_pattern,
            num_generations=num_generations,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    @classmethod
    def from_llamacpp(
        cls,
        llm: "LlamaCppLLM",
        sampler: SamplerType = "multinomial",
        output_format: OutputType = "text",
        output_structure: Optional[StructureType] = None,
        whitespace_pattern: Optional[str] = None,
        num_generations: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Self:
        """Creates an `OutlinesStructuredOutput` from a `LlamaCppLLM`.

        Args:
            llm: The `LlamaCppLLM` instance to use for the generation.
            sampler: the sampler for the logits. Defaults to "multinomial".
            output_format: Structured output wanted from the `LLM`. Defaults to "text".
            output_structure: the structure of the output. Defaults to `None`.
            whitespace_pattern: the pattern to use to split the output into structured parts. Defaults to `None`.
            num_generations: the number of generations to produce. Defaults to `1`.
            top_k: the number of top tokens to sample from. Defaults to `None`.
            top_p: the cumulative probability threshold for sampling from the top tokens. Defaults to `None`.
            temperature: the temperature to use for sampling. Defaults to `None`.

        Returns:
            `OutlinesStructuredOutput` instance.
        """
        from outlines.models.llamacpp import LlamaCpp

        return cls(
            llm=LlamaCpp(llm._model),
            sampler=sampler,
            output_format=output_format,
            output_structure=output_structure,
            whitespace_pattern=whitespace_pattern,
            num_generations=num_generations,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    @classmethod
    def from_vllm(
        cls,
        llm: "vLLM",
        sampler: SamplerType = "multinomial",
        output_format: OutputType = "text",
        output_structure: Optional[StructureType] = None,
        whitespace_pattern: Optional[str] = None,
        num_generations: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Self:
        """Creates an `OutlinesStructuredOutput` from a `vLLM`.

        Args:
            llm: The `vLLM` instance to use for the generation.
            sampler: the sampler for the logits. Defaults to "multinomial".
            output_format: Structured output wanted from the `LLM`. Defaults to "text".
            output_structure: the structure of the output. Defaults to `None`.
            whitespace_pattern: the pattern to use to split the output into structured parts. Defaults to `None`.
            num_generations: the number of generations to produce. Defaults to `1`.
            top_k: the number of top tokens to sample from. Defaults to `None`.
            top_p: the cumulative probability threshold for sampling from the top tokens. Defaults to `None`.
            temperature: the temperature to use for sampling. Defaults to `None`.

        Returns:
            `OutlinesStructuredOutput` instance.
        """
        from outlines.models.vllm import VLLM

        return cls(
            llm=VLLM(llm._model),
            sampler=sampler,
            output_format=output_format,
            output_structure=output_structure,
            whitespace_pattern=whitespace_pattern,
            num_generations=num_generations,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    @classmethod
    def from_openai(
        cls,
        llm: Union["OpenAILLM", "AzureOpenAILLM"],
        sampler: SamplerType = "multinomial",
        output_format: OutputType = "text",
        output_structure: Optional[StructureType] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        num_generations: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Self:
        """Creates an `OutlinesStructuredOutput` from a `OpenAILLM` or `AzureOpenAILLM`.

        Args:
            llm: the `OpenAILLM` or `AzureOpenAILLM` instance to use for the generation.
            sampler: the sampler for the logits. Defaults to "multinomial".
            output_format: Structured output wanted from the `LLM`. Defaults to "text".
            output_structure: the structure of the output. Defaults to `None`.
            stop_at: a string or a list of strings to use as a stop sequence for the generation.
                Defaults to `None`.
            num_generations: the number of generations to produce. Defaults to `1`.
            top_k: the number of top tokens to sample from. Defaults to `None`.
            top_p: the cumulative probability threshold for sampling from the top tokens. Defaults to `None`.
            temperature: the temperature to use for sampling. Defaults to `None`.
            seed: optional integer to seed the generation. Defaults to None.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            presence_penalty the presence penalty to use for the generation. Defaults to
                `0.0`.
            max_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.

        Returns:
            `OutlinesStructuredOutput` instance.
        """
        if output_format in ("json", "regex", "cfg"):
            raise NotImplementedError(
                f"Only 'text' output_format is supported for `{type(llm).__name__}`."
            )
        import tiktoken
        from outlines.models.openai import OpenAI, OpenAIConfig

        config = OpenAIConfig(
            model=llm.model_name,
            seed=seed,
            stop=stop_at,
            # Arguments that would fit in the sampler in other cases, or generation_kwargs.
            temperature=temperature,
            top_p=top_p,
            n=num_generations,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        llm = OpenAI(
            llm._aclient,  # client
            config=config,
            tokenizer=tiktoken.encoding_for_model(llm.model_name),
        )

        return cls(
            llm=llm,
            sampler=sampler,
            output_format=output_format,
            output_structure=output_structure,
            num_generations=num_generations,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    @classmethod
    def _from_llm(
        cls,
        llm: AllowedLLMs,
        **kwargs: Any,
    ) -> Self:
        """Convenient method to create the `OutlinesStructuredOutput` from any allowed `LLM`.

        It's intended for internal use, go the specific method for the `LLM` you want to use.

        Args:
            llm: the `LLM` instance to use for the generation.
            **kwargs: the arguments to pass to the specific method for the `LLM`.

        Raises:
            NotImplementedError: for `LLM` that are not in the `AllowedLLMs`.

        Returns:
            `OutlinesStructuredOutput` instance.
        """
        # hackish way to check the type of the llm without importing it the modules,
        # to avoid raising an ImportError just to check the framework used.
        llm_name = type(llm).__name__.lower()

        if "transformers" in llm_name:
            return cls.from_transformers(llm=llm, **kwargs)
        elif "llamacpp" in llm_name:
            return cls.from_llamacpp(llm=llm, **kwargs)
        elif "vllm" in llm_name:
            return cls.from_vllm(llm=llm, **kwargs)
        elif "openai" in llm_name:
            return cls.from_openai(llm=llm, **kwargs)
        raise NotImplementedError(
            f"Only {get_args(AllowedLLMs)} are supported for {type(cls).__name__}."
        )

    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
    ) -> List["GenerateOutput"]:
        """Metho to effectively generate the structured outputs from the `LLM`.

        Args:
            prompts: the prompts to generate the structured outputs (inputs in the `LLMs`).
            max_tokens: the maximum number of tokens to generate. Defaults to None.
            stop_at: a string or a list of strings to use as a stop sequence for the generation.
        """

        # NOTE: rng isn't passed for the moment, we need a better way of creating it thinking of possibly serializing the value.
        try:
            structured_output = self._structured_generator(
                prompts, max_tokens=max_tokens, stop_at=stop_at
            )
        except json.decoder.JSONDecodeError:
            # If the model is not strong enough, or the max_tokens
            # is too low, the output can be a string that is not a valid JSON.
            logger = logging.getLogger(
                "distilabel.steps.tasks.structured_outputs.outlines"
            )
            logger.warning(
                "Error decoding the JSON structured output. Returning empty dict."
            )
            structured_output = "{}"
        except Exception as e:
            logger = logging.getLogger(
                "distilabel.steps.tasks.structured_outputs.outlines"
            )
            logger.warning(
                f"Error decoding the structured output. Returning empty str. Error: {e}"
            )
            structured_output = ""

        if isinstance(structured_output, list):
            if isinstance(structured_output[0], list):
                # num_generations > 1, batch_size > 1.
                outputs = structured_output
            else:
                # num_generations or batch_size > 1, the other == 1.
                outputs = [structured_output]
        else:
            # When num_generations==1, batch_size==1
            outputs = [[structured_output]]

        if self.output_format == "json":
            # If the output is in json format, pass it as a string to not break latter steps in the pipeline.
            return [[json.dumps(output) for output in batch] for batch in outputs]
        return outputs

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        # Just don't include the llm in the dump as it needs special treatment.
        return self.model_dump(exclude="llm", **kwargs)
