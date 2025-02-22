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

import contextlib
import gc
import inspect
import json
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from pydantic import Field, PrivateAttr, SecretStr, validate_call

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.base import LLM
from distilabel.models.llms.openai import OpenAILLM
from distilabel.models.llms.utils import compute_tokens, prepare_output
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.structured_outputs.utils import schema_as_dict
from distilabel.typing import (
    FormattedInput,
    GenerateOutput,
    Logprob,
    OutlinesStructuredOutputType,
)

if TYPE_CHECKING:
    from openai import OpenAI  # noqa
    from transformers import PreTrainedTokenizer
    from sglang import Engine as _SGLang

    from distilabel.typing import (
        StandardInput,
        StructuredInput,
        LLMStatistics,
        LLMLogprobs,
        LLMOutput,
    )


LogitsProcessorFn = Union[
    Callable[[List[int], Any], Any],
    Callable[[List[int], List[int], Any], Any],
]

LogitsProcessors = List[LogitsProcessorFn]


class SGLang(LLM, MagpieChatTemplateMixin, CudaDevicePlacementMixin):
    """`SGLang` library LLM implementation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        dtype: the data type to use for the model. Defaults to `auto`.
        trust_remote_code: whether to trust the remote code when loading the model. Defaults
            to `False`.
        quantization: the quantization mode to use for the model. Defaults to `None`.
        revision: the revision of the model to load. Defaults to `None`.
        tokenizer: the tokenizer Hugging Face Hub repo id or a path to a directory containing
            the tokenizer files. If not provided, the tokenizer will be loaded from the
            model directory. Defaults to `None`.
        tokenizer_mode: the mode to use for the tokenizer. Defaults to `auto`.
        tokenizer_revision: the revision of the tokenizer to load. Defaults to `None`.
        skip_tokenizer_init: whether to skip the initialization of the tokenizer. Defaults
            to `False`.
        chat_template: a chat template that will be used to build the prompts before
            sending them to the model. If not provided, the chat template defined in the
            tokenizer config will be used. If not provided and the tokenizer doesn't have
            a chat template, then ChatML template will be used. Defaults to `None`.
        structured_output: a dictionary containing the structured output configuration or if more
            fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.
        seed: the seed to use for the random number generator. Defaults to `0`.
        extra_kwargs: additional dictionary of keyword arguments that will be passed to the
            `LLM` class of `vllm` library. Defaults to `{}`.
        _model: the `vLLM` model instance. This attribute is meant to be used internally
            and should not be accessed directly. It will be set in the `load` method.
        _tokenizer: the tokenizer instance used to format the prompt before passing it to
            the `LLM`. This attribute is meant to be used internally and should not be
            accessed directly. It will be set in the `load` method.
        use_magpie_template: a flag used to enable/disable applying the Magpie pre-query
            template. Defaults to `False`.
        magpie_pre_query_template: the pre-query template to be applied to the prompt or
            sent to the LLM to generate an instruction or a follow up user message. Valid
            values are "llama3", "qwen2" or another pre-query template provided. Defaults
            to `None`.

    References:
        - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py

    Runtime parameters:
        - `extra_kwargs`: additional dictionary of keyword arguments that will be passed to
            the `LLM` class of `SGLang` library.

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import vLLM

        # You can pass a custom chat_template to the model
        llm = vLLM(
            model="prometheus-eval/prometheus-7b-v2.0",
            chat_template="[INST] {{ messages[0]\"content\" }}\\n{{ messages[1]\"content\" }}[/INST]",
        )

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Generate structured data:

        ```python
        from pathlib import Path
        from distilabel.models.llms import vLLM

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = vLLM(
            model="prometheus-eval/prometheus-7b-v2.0"
            structured_output={"format": "json", "schema": Character},
        )

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
        ```
    """

    model: str
    dtype: str = "auto"
    trust_remote_code: bool = False
    quantization: Optional[str] = None
    revision: Optional[str] = None

    tokenizer: Optional[str] = None
    tokenizer_mode: Literal["auto", "slow"] = "auto"
    skip_tokenizer_init: bool = False
    chat_template: Optional[str] = None

    seed: int = 0

    extra_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `vLLM` class of `vllm` library. See all the supported arguments at: "
        "https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py",
    )
    structured_output: Optional[RuntimeParameter[OutlinesStructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )

    _model: "_SGLang" = PrivateAttr(None)
    _tokenizer: "PreTrainedTokenizer" = PrivateAttr(None)
    # Jayon: I don't know
    _structured_output_logits_processor: Optional[Callable] = PrivateAttr(default=None)

    def load(self) -> None:
        """Loads the `vLLM` model using either the path or the Hugging Face Hub repository id.
        Additionally, this method also sets the `chat_template` for the tokenizer, so as to properly
        parse the list of OpenAI formatted inputs using the expected format by the model, otherwise, the
        default value is ChatML format, unless explicitly provided.
        """
        super().load()

        CudaDevicePlacementMixin.load(self)

        try:
            from sglang import Engine as _SGLang
        except ImportError as err:
            raise ImportError(
                "sglang is not installed. Please install it with sglang document."
            ) from err

        self._model = _SGLang(
            model_path=self.model,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            revision=self.revision,
            tokenizer_path=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            skip_tokenizer_init=self.skip_tokenizer_init,
            random_seed=self.seed,
            **self.extra_kwargs,  # type: ignore
        )
        from sglang.srt.hf_transformers_utils import get_tokenizer

        self._tokenizer = get_tokenizer(
            self.model,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            tokenizer_revision="main",
        )  # type: ignore
        if self.chat_template is not None:
            self._tokenizer.chat_template = self.chat_template  # type: ignore

        # if self.structured_output:
        #     self._structured_output_logits_processor = self._prepare_structured_output(
        #         self.structured_output
        #     )

    def unload(self) -> None:
        """Unloads the `SGLang` model."""
        self._cleanup_sglang_model()
        self._model = None  # type: ignore
        self._tokenizer = None  # type: ignore
        CudaDevicePlacementMixin.unload(self)
        super().unload()

    def _cleanup_sglang_model(self) -> None:
        if self._model is None:
            return

        import torch  # noqa

        self._model.shutdown()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    def prepare_input(self, input: Union["StandardInput", str]) -> str:
        """Prepares the input (applying the chat template and tokenization) for the provided
        input.

        Args:
            input: the input list containing chat items.

        Returns:
            The prompt to send to the LLM.
        """
        if isinstance(input, str):
            return input

        prompt: str = (
            self._tokenizer.apply_chat_template(
                input,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,  # type: ignore
            )
            if input
            else ""
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    def _prepare_batches(
        self, inputs: List["StructuredInput"]
    ) -> Tuple[List[Tuple[List[str], "OutlinesStructuredOutputType"]], List[int]]:
        """Prepares the inputs by grouping them by the structured output.

        When we generate structured outputs with schemas obtained from a dataset, we need to
        prepare the data to try to send batches of inputs instead of single inputs to the model
        to take advante of the engine. So we group the inputs by the structured output to be
        passed in the `generate` method.

        Args:
            inputs: The batch of inputs passed to the generate method. As we expect to be generating
                structured outputs, each element will be a tuple containing the instruction and the
                structured output.

        Returns:
            The prepared batches (sub-batches let's say) to be passed to the `generate` method.
            Each new tuple will contain instead of the single instruction, a list of instructions
        """
        instruction_order = {}
        batches: Dict[str, List[str]] = {}
        for i, (instruction, structured_output) in enumerate(inputs):
            instruction = self.prepare_input(instruction)
            instruction_order[instruction] = i

            structured_output = json.dumps(structured_output)
            if structured_output not in batches:
                batches[structured_output] = [instruction]
            else:
                batches[structured_output].append(instruction)

        # Built a list with instructions sorted by structured output
        flat_instructions = [
            instruction for _, group in batches.items() for instruction in group
        ]

        # Generate the list of indices based on the original order
        sorted_indices = [
            instruction_order[instruction] for instruction in flat_instructions
        ]

        return [
            (batch, json.loads(schema)) for schema, batch in batches.items()
        ], sorted_indices

    @validate_call
    def generate(  # noqa: C901 # type: ignore
        self,
        inputs: List[FormattedInput],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        spaces_between_special_tokens: bool = True,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        no_stop_trim: bool = False,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        custom_params: Optional[Dict[str, Any]] = None,  # no use?
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        echo: bool = False,
    ) -> List[GenerateOutput]:
        """Generates `num_generations` responses for each input.

        Args:
            inputs: a list of inputs in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            repetition_penalty: the repetition penalty to use for the generation Defaults to
                `1.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            top_k: the top-k value to use for the generation. Defaults to `0`.
            min_p: the minimum probability to use for the generation. Defaults to `0.0`.
            logprobs: number of log probabilities to return per output token. If `None`,
                then no log probability won't be returned. Defaults to `None`.
            stop: a list of strings that will be used to stop the generation when found.
                Defaults to `None`.
            stop_token_ids: a list of token ids that will be used to stop the generation
                when found. Defaults to `None`.
            include_stop_str_in_output: whether to include the stop string in the output.
                Defaults to `False`.
            skip_special_tokens: whether to exclude special tokens from the output. Defaults
                to `False`.
            logits_processors: a list of functions to process the logits before sampling.
                Defaults to `None`.
            extra_sampling_params: dictionary with additional arguments to be passed to
                the `SamplingParams` class from `vllm`.
            echo: whether to echo the include the prompt in the response or not. Defaults
                to `False`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        if isinstance(inputs[0], tuple):
            # Prepare the batches for structured generation
            prepared_batches, sorted_indices = self._prepare_batches(inputs)  # type: ignore
        else:
            # Simulate a batch without the structured output content
            prepared_batches = [([self.prepare_input(input) for input in inputs], None)]  # type: ignore
            sorted_indices = None

        batched_outputs: List["LLMOutput"] = []
        generations = []
        # import pdb
        # pdb.set_trace()
        for prepared_inputs, structured_output in prepared_batches:
            if self.structured_output is not None and structured_output is not None:
                self._logger.warning(
                    "An `structured_output` was provided in the model configuration, but"
                    " one was also provided in the input. The input structured output will"
                    " be used."
                )

            temp_structure = None
            if structured_output is not None:
                temp_structure = structured_output
            elif self.structured_output is not None:
                temp_structure = self.structured_output

            if temp_structure is not None:
                format = temp_structure.get("format")
                schema = temp_structure.get("schema")
                if not format:
                    if isinstance(schema, dict) or inspect.isclass(schema):
                        format = "json"
                    elif isinstance(schema, str):
                        format = "regex"

                if format == "json":
                    json_schema = json.dumps(schema_as_dict(schema))
                elif format == "regex":
                    regex = schema

            sampling_params = {
                "max_new_tokens": max_new_tokens,
                "stop": stop,
                "stop_token_ids": stop_token_ids,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
                "min_new_tokens": min_new_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens,
                "n": num_generations,
                "json_schema": json_schema,
                "regex": regex,
                "ebnf": ebnf,
                "no_stop_trim": no_stop_trim,
                "ignore_eos": ignore_eos,
                "skip_special_tokens": skip_special_tokens,
                "custom_params": custom_params,
            }

            batch_outputs = self._model.generate(
                prompt=prepared_inputs,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                logprob_start_len=0 if echo else -1,
                top_logprobs_num=top_logprobs_num if return_logprob else 0,
            )

            for input, outputs in zip(prepared_inputs, batch_outputs):
                processed_prompt_logprobs = []
                meta_info = outputs["meta_info"]
                if "input_top_logprobs" in meta_info:
                    processed_prompt_logprobs = self._get_llm_logprobs(
                        top_logprob=meta_info["input_top_logprobs"],
                        choose_logprob=meta_info["input_token_logprobs"],
                    )
                texts, statistics, outputs_logprobs = self._process_outputs(
                    input=input,
                    outputs=outputs,
                    echo=echo,
                    prompt_logprobs=processed_prompt_logprobs,
                )
                batched_outputs.append(texts)
                generation = prepare_output(
                    generations=texts,
                    input_tokens=statistics["input_tokens"],
                    output_tokens=statistics["output_tokens"],
                    logprobs=outputs_logprobs,
                )

                generations.append(generation)

        if sorted_indices is not None:
            pairs = list(enumerate(sorted_indices))
            pairs.sort(key=lambda x: x[1])
            generations = [generations[original_idx] for original_idx, _ in pairs]

        return generations

    def _process_outputs(
        self,
        input: str,
        outputs,
        prompt_logprobs: List[List["Logprob"]],
        echo: bool = False,
    ) -> Tuple["LLMOutput", "LLMStatistics", "LLMLogprobs"]:
        texts = []
        outputs_logprobs = []
        lens = 1
        if isinstance(outputs, list):
            lens = len(outputs)
        statistics = {
            "input_tokens": [compute_tokens(input, self._tokenizer.encode)] * lens,
            "output_tokens": [],
        }
        if lens == 1:
            text = outputs["text"]
            if echo:
                text = input + text
            texts.append(text)
            statistics["output_tokens"].append(
                outputs["meta_info"]["completion_tokens"]
            )
            if "output_top_logprobs" in outputs["meta_info"]:
                processed_output_logprobs = self._get_llm_logprobs(
                    outputs["meta_info"]["output_top_logprobs"]
                )
                outputs_logprobs.append(prompt_logprobs + processed_output_logprobs)
        else:
            raise ValueError("lens is not 1 when _process_outputs")
        return texts, statistics, outputs_logprobs

    # def _prepare_structured_output(  # type: ignore
    #     self, structured_output: "OutlinesStructuredOutputType"
    # ) -> Union[Callable, None]:
    #     """Creates the appropriate function to filter tokens to generate structured outputs.

    #     Args:
    #         structured_output: the configuration dict to prepare the structured output.

    #     Returns:
    #         The callable that will be used to guide the generation of the model.
    #     """
    #     from distilabel.steps.tasks.structured_outputs.outlines import (
    #         prepare_guided_output,
    #     )

    #     assert structured_output is not None, "`structured_output` cannot be `None`"

    #     result = prepare_guided_output(structured_output, "sglang", self._model)
    #     if (schema := result.get("schema")) and self.structured_output:
    #         self.structured_output["schema"] = schema
    #     return result["processor"]

    def _get_llm_logprobs(
        self,
        top_logprob,
        choose_logprob=None,
    ) -> List[List["Logprob"]]:
        processed_logprobs = []
        if choose_logprob is not None:
            token_logprobs = []
            for num in range(len(choose_logprob)):
                if choose_logprob[num][0] is None:
                    processed_logprobs.append(None)
                    continue
                else:
                    token_logprobs.append(
                        {
                            "token": choose_logprob[num][2],
                            "logprob": choose_logprob[num][0],
                        }
                    )
                for top_num in range(len(top_logprob[num]) - 1):
                    token_logprobs.append(
                        {
                            "token": top_logprob[num][top_num][2],
                            "logprob": top_logprob[num][top_num][0],
                        }
                    )
            processed_logprobs.append(token_logprobs)
        else:
            for probs in top_logprob:
                token_logprobs = []
                for item in probs:
                    token_logprobs.append({"token": item[2], "logprob": item[0]})
                processed_logprobs.append(token_logprobs)

        return processed_logprobs


class ClientSGLang(OpenAILLM, MagpieChatTemplateMixin):
    """A client for the `SGLang` server implementing the OpenAI API specification.

    Attributes:
        base_url: the base URL of the `SGLang` server. Defaults to `"http://localhost:30000"`.
        max_retries: the maximum number of times to retry the request to the API before
            failing. Defaults to `6`.
        timeout: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        httpx_client_kwargs: extra kwargs that will be passed to the `httpx.AsyncClient`
            created to comunicate with the `vLLM` server. Defaults to `None`.
        tokenizer: the Hugging Face Hub repo id or path of the tokenizer that will be used
            to apply the chat template and tokenize the inputs before sending it to the
            server. Defaults to `None`.
        tokenizer_revision: the revision of the tokenizer to load. Defaults to `None`.
        _aclient: the `httpx.AsyncClient` used to comunicate with the `vLLM` server. Defaults
            to `None`.

    Runtime parameters:
        - `base_url`: the base url of the `vLLM` server. Defaults to `"http://localhost:30000"`.
        - `max_retries`: the maximum number of times to retry the request to the API before
            failing. Defaults to `6`.
        - `timeout`: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        - `httpx_client_kwargs`: extra kwargs that will be passed to the `httpx.AsyncClient`
            created to comunicate with the `vLLM` server. Defaults to `None`.

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import ClientvLLM

        llm = ClientvLLM(
            base_url="http://localhost:30000/v1",
            tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct"
        )

        llm.load()

        results = llm.generate_outputs(
            inputs=[[{"role": "user", "content": "Hello, how are you?"}]],
            temperature=0.7,
            top_p=1.0,
            max_new_tokens=256,
        )
        # [
        #     [
        #         "I'm functioning properly, thank you for asking. How can I assist you today?",
        #         "I'm doing well, thank you for asking. I'm a large language model, so I don't have feelings or emotions like humans do, but I'm here to help answer any questions or provide information you might need. How can I assist you today?",
        #         "I'm just a computer program, so I don't have feelings like humans do, but I'm functioning properly and ready to help you with any questions or tasks you have. What's on your mind?"
        #     ]
        # ]
        ```
    """

    model: str = ""  # Default value so it's not needed to `ClientvLLM(model="...")`
    tokenizer: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    # We need the sync client to get the list of models
    _client: "OpenAI" = PrivateAttr(None)
    _tokenizer: "PreTrainedTokenizer" = PrivateAttr(None)

    def load(self) -> None:
        """Creates an `httpx.AsyncClient` to connect to the vLLM server and a tokenizer
        optionally."""

        self.api_key = SecretStr("EMPTY")

        # We need to first create the sync client to get the model name that will be used
        # in the `super().load()` when creating the logger.
        try:
            from openai import OpenAI
        except ImportError as ie:
            raise ImportError(
                "OpenAI Python client is not installed. Please install it using"
                " `pip install 'distilabel[openai]'`."
            ) from ie

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),  # type: ignore
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,
        )

        super().load()

        try:
            from transformers import AutoTokenizer
        except ImportError as ie:
            raise ImportError(
                "To use `ClientvLLM` you need to install `transformers`."
                "Please install it using `pip install 'distilabel[hf-transformers]'`."
            ) from ie

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer, revision=self.tokenizer_revision
        )

    @cached_property
    def model_name(self) -> str:  # type: ignore
        """Returns the name of the model served with vLLM server."""
        models = self._client.models.list()
        return models.data[0].id

    def _prepare_input(self, input: "StandardInput") -> str:
        """Prepares the input (applying the chat template and tokenization) for the provided
        input.

        Args:
            input: the input list containing chat items.

        Returns:
            The prompt to send to the LLM.
        """
        prompt: str = (
            self._tokenizer.apply_chat_template(  # type: ignore
                input,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,  # type: ignore
            )
            if input
            else ""
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, int]] = None,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> GenerateOutput:
        """Generates `num_generations` responses for each input.

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            logit_bias: modify the likelihood of specified tokens appearing in the completion.
                Defaults to ``
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: nucleus sampling. The value refers to the top-p tokens that should be
                considered for sampling. Defaults to `1.0`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        completion = await self._aclient.completions.create(
            model=self.model_name,
            prompt=self._prepare_input(input),  # type: ignore
            n=num_generations,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
        )

        generations = []
        for choice in completion.choices:
            text = choice.text
            if text == "":
                self._logger.warning(  # type: ignore
                    f"Received no response from SGLang server (model: '{self.model_name}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(text)

        return prepare_output(generations, **self._get_llm_statistics(completion))
