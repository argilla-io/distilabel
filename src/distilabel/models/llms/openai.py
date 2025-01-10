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

import io
import os
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Union

import orjson
from pydantic import Field, PositiveInt, PrivateAttr, SecretStr, validate_call

from distilabel import envs
from distilabel.exceptions import DistilabelOfflineBatchGenerationNotFinishedException
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.base import AsyncLLM
from distilabel.models.llms.typing import GenerateOutput
from distilabel.models.llms.utils import prepare_output
from distilabel.steps.tasks.typing import FormattedInput, InstructorStructuredOutputType

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openai.types import Batch as OpenAIBatch
    from openai.types import FileObject as OpenAIFileObject
    from openai.types.chat import ChatCompletion as OpenAIChatCompletion
    from openai.types.chat.chat_completion import Choice as OpenAIChoice
    from openai.types.completion import Completion as OpenAICompletion

    from distilabel.models.llms.typing import LLMStatistics, Logprob


_OPENAI_API_KEY_ENV_VAR_NAME = "OPENAI_API_KEY"
_OPENAI_BATCH_API_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


class OpenAILLM(AsyncLLM):
    """OpenAI LLM implementation running the async API client.

    Attributes:
        model: the model name to use for the LLM e.g. "gpt-3.5-turbo", "gpt-4", etc.
            Supported models can be found [here](https://platform.openai.com/docs/guides/text-generation).
        base_url: the base URL to use for the OpenAI API requests. Defaults to `None`, which
            means that the value set for the environment variable `OPENAI_BASE_URL` will
            be used, or "https://api.openai.com/v1" if not set.
        api_key: the API key to authenticate the requests to the OpenAI API. Defaults to
            `None` which means that the value set for the environment variable `OPENAI_API_KEY`
            will be used, or `None` if not set.
        default_headers: the default headers to use for the OpenAI API requests.
        max_retries: the maximum number of times to retry the request to the API before
            failing. Defaults to `6`.
        timeout: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        structured_output: a dictionary containing the structured output configuration configuration
            using `instructor`. You can take a look at the dictionary structure in
            `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.

    Runtime parameters:
        - `base_url`: the base URL to use for the OpenAI API requests. Defaults to `None`.
        - `api_key`: the API key to authenticate the requests to the OpenAI API. Defaults
            to `None`.
        - `max_retries`: the maximum number of times to retry the request to the API before
            failing. Defaults to `6`.
        - `timeout`: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.

    Icon:
        `:simple-openai:`

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import OpenAILLM

        llm = OpenAILLM(model="gpt-4-turbo", api_key="api.key")

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Generate text from a custom endpoint following the OpenAI API:

        ```python
        from distilabel.models.llms import OpenAILLM

        llm = OpenAILLM(
            model="prometheus-eval/prometheus-7b-v2.0",
            base_url=r"http://localhost:8080/v1"
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Generate structured data:

        ```python
        from pydantic import BaseModel
        from distilabel.models.llms import OpenAILLM

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = OpenAILLM(
            model="gpt-4-turbo",
            api_key="api.key",
            structured_output={"schema": User}
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
        ```

        Generate with Batch API (offline batch generation):

        ```python
        from distilabel.models.llms import OpenAILLM

        load = llm = OpenAILLM(
            model="gpt-3.5-turbo",
            use_offline_batch_generation=True,
            offline_batch_generation_block_until_done=5,  # poll for results every 5 seconds
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        # [['Hello! How can I assist you today?']]
        ```
    """

    model: str
    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ),
        description="The base URL to use for the OpenAI API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_OPENAI_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the OpenAI API.",
    )
    default_headers: Optional[RuntimeParameter[Dict[str, str]]] = Field(
        default=None,
        description="The default headers to use for the OpenAI API requests.",
    )
    max_retries: RuntimeParameter[int] = Field(
        default=6,
        description="The maximum number of times to retry the request to the API before"
        " failing.",
    )
    timeout: RuntimeParameter[int] = Field(
        default=120,
        description="The maximum time in seconds to wait for a response from the API.",
    )
    structured_output: Optional[RuntimeParameter[InstructorStructuredOutputType]] = (
        Field(
            default=None,
            description="The structured output format to use across all the generations.",
        )
    )

    _api_key_env_var: str = PrivateAttr(_OPENAI_API_KEY_ENV_VAR_NAME)
    _client: "OpenAI" = PrivateAttr(None)
    _aclient: "AsyncOpenAI" = PrivateAttr(None)

    def load(self) -> None:
        """Loads the `AsyncOpenAI` client to benefit from async requests."""
        super().load()

        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as ie:
            raise ImportError(
                "OpenAI Python client is not installed. Please install it using"
                " `pip install 'distilabel[openai]'`."
            ) from ie

        if self.api_key is None:
            raise ValueError(
                f"To use `{self.__class__.__name__}` an API key must be provided via `api_key`"
                f" attribute or runtime parameter, or set the environment variable `{self._api_key_env_var}`."
            )

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,
            default_headers=self.default_headers,
        )

        self._aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,
            default_headers=self.default_headers,
        )

        if self.structured_output:
            result = self._prepare_structured_output(
                structured_output=self.structured_output,
                client=self._aclient,
                framework="openai",
            )
            self._aclient = result.get("client")  # type: ignore
            if structured_output := result.get("structured_output"):
                self.structured_output = structured_output

    def unload(self) -> None:
        """Set clients to `None` as they both contain `thread._RLock` which cannot be pickled
        in case an exception is raised and has to be handled in the main process"""

        self._client = None  # type: ignore
        self._aclient = None  # type: ignore
        self.default_headers = None
        self.structured_output = None
        super().unload()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        logprobs: bool = False,
        top_logprobs: Optional[PositiveInt] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> GenerateOutput:
        """Generates `num_generations` responses for the given input using the OpenAI async
        client.

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            logprobs: whether to return the log probabilities or not. Defaults to `False`.
            top_logprobs: the number of top log probabilities to return per output token
                generated. Defaults to `None`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            stop: a string or a list of strings to use as a stop sequence for the generation.
                Defaults to `None`.
            response_format: the format of the response to return. Must be one of
                "text" or "json". Read the documentation [here](https://platform.openai.com/docs/guides/text-generation/json-mode)
                for more information on how to use the JSON model from OpenAI. Defaults to None
                which returns text. To return JSON, use {"type": "json_object"}.

        Note:
            If response_format

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        structured_output = None
        if isinstance(input, tuple):
            input, structured_output = input
            result = self._prepare_structured_output(
                structured_output=structured_output,  # type: ignore
                client=self._aclient,
                framework="openai",
            )
            self._aclient = result.get("client")  # type: ignore

        if structured_output is None and self.structured_output is not None:
            structured_output = self.structured_output

        kwargs = {
            "messages": input,  # type: ignore
            "model": self.model,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "max_tokens": max_new_tokens,
            "n": num_generations,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        # Check if it's a vision generation task, in that case "stop" cannot be used or raises
        # an error in the API.
        if isinstance(
            [row for row in input if row["role"] == "user"][0]["content"], list
        ):
            kwargs.pop("stop")

        if response_format is not None:
            kwargs["response_format"] = response_format

        if structured_output:
            kwargs = self._prepare_kwargs(kwargs, structured_output)  # type: ignore

        completion = await self._aclient.chat.completions.create(**kwargs)  # type: ignore

        if structured_output:
            # NOTE: `instructor` doesn't work with `n` parameter, so it will always return
            # only 1 choice.
            statistics = self._get_llm_statistics(completion._raw_response)
            if choice_logprobs := self._get_logprobs_from_choice(
                completion._raw_response.choices[0]
            ):
                output_logprobs = [choice_logprobs]
            else:
                output_logprobs = None
            return prepare_output(
                generations=[completion.model_dump_json()],
                input_tokens=statistics["input_tokens"],
                output_tokens=statistics["output_tokens"],
                logprobs=output_logprobs,
            )

        return self._generations_from_openai_completion(completion)

    def _generations_from_openai_completion(
        self, completion: "OpenAIChatCompletion"
    ) -> "GenerateOutput":
        """Get the generations from the OpenAI Chat Completion object.

        Args:
            completion: the completion object to get the generations from.

        Returns:
            A list of strings containing the generated responses for the input.
        """
        generations = []
        logprobs = []
        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(  # type: ignore
                    f"Received no response using OpenAI client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
            if choice_logprobs := self._get_logprobs_from_choice(choice):
                logprobs.append(choice_logprobs)

        statistics = self._get_llm_statistics(completion)
        return prepare_output(
            generations=generations,
            input_tokens=statistics["input_tokens"],
            output_tokens=statistics["output_tokens"],
            logprobs=logprobs,
        )

    def _get_logprobs_from_choice(
        self, choice: "OpenAIChoice"
    ) -> Union[List[List["Logprob"]], None]:
        if choice.logprobs is None or choice.logprobs.content is None:
            return None

        return [
            [
                {"token": top_logprob.token, "logprob": top_logprob.logprob}
                for top_logprob in token_logprobs.top_logprobs
            ]
            for token_logprobs in choice.logprobs.content
        ]

    def offline_batch_generate(
        self,
        inputs: Union[List["FormattedInput"], None] = None,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        logprobs: bool = False,
        top_logprobs: Optional[PositiveInt] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        """Uses the OpenAI batch API to generate `num_generations` responses for the given
        inputs.

        Args:
            inputs: a list of inputs in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            logprobs: whether to return the log probabilities or not. Defaults to `False`.
            top_logprobs: the number of top log probabilities to return per output token
                generated. Defaults to `None`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            stop: a string or a list of strings to use as a stop sequence for the generation.
                Defaults to `None`.
            response_format: the format of the response to return. Must be one of
                "text" or "json". Read the documentation [here](https://platform.openai.com/docs/guides/text-generation/json-mode)
                for more information on how to use the JSON model from OpenAI. Defaults to `text`.

        Returns:
            A list of lists of strings containing the generated responses for each input
            in `inputs`.

        Raises:
            DistilabelOfflineBatchGenerationNotFinishedException: if the batch generation
                is not finished yet.
            ValueError: if no job IDs were found to retrieve the results from.
        """
        if self.jobs_ids:
            return self._check_and_get_batch_results()

        if inputs:
            self.jobs_ids = self._create_jobs(
                inputs=inputs,
                **{
                    "model": self.model,
                    "logprobs": logprobs,
                    "top_logprobs": top_logprobs,
                    "max_tokens": max_new_tokens,
                    "n": num_generations,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop,
                    "response_format": response_format,
                },
            )
            raise DistilabelOfflineBatchGenerationNotFinishedException(
                jobs_ids=self.jobs_ids
            )

        raise ValueError("No `inputs` were provided and no `jobs_ids` were found.")

    def _check_and_get_batch_results(self) -> List["GenerateOutput"]:
        """Checks the status of the batch jobs and retrieves the results from the OpenAI
        Batch API.

        Returns:
            A list of lists of strings containing the generated responses for each input.

        Raises:
            ValueError: if no job IDs were found to retrieve the results from.
            DistilabelOfflineBatchGenerationNotFinishedException: if the batch generation
                is not finished yet.
            RuntimeError: if the only batch job found failed.
        """
        if not self.jobs_ids:
            raise ValueError("No job IDs were found to retrieve the results from.")

        outputs = []
        for batch_id in self.jobs_ids:
            batch = self._get_openai_batch(batch_id)

            if batch.status in ("validating", "in_progress", "finalizing"):
                raise DistilabelOfflineBatchGenerationNotFinishedException(
                    jobs_ids=self.jobs_ids
                )

            if batch.status in ("failed", "expired", "cancelled", "cancelling"):
                self._logger.error(  # type: ignore
                    f"OpenAI API batch with ID '{batch_id}' failed with status '{batch.status}'."
                )
                if len(self.jobs_ids) == 1:
                    self.jobs_ids = None
                    raise RuntimeError(
                        f"The only OpenAI API Batch that was created with ID '{batch_id}'"
                        f" failed with status '{batch.status}'."
                    )

                continue

            outputs.extend(self._retrieve_batch_results(batch))

        # sort by `custom_id` to return the results in the same order as the inputs
        outputs = sorted(outputs, key=lambda x: int(x["custom_id"]))
        return [self._parse_output(output) for output in outputs]

    def _parse_output(self, output: Dict[str, Any]) -> "GenerateOutput":
        """Parses the output from the OpenAI Batch API into a list of strings.

        Args:
            output: the output to parse.

        Returns:
            A list of strings containing the generated responses for the input.
        """
        from openai.types.chat import ChatCompletion as OpenAIChatCompletion

        if "response" not in output:
            return []

        if output["response"]["status_code"] != 200:
            return []

        return self._generations_from_openai_completion(
            OpenAIChatCompletion(**output["response"]["body"])
        )

    def _get_openai_batch(self, batch_id: str) -> "OpenAIBatch":
        """Gets a batch from the OpenAI Batch API.

        Args:
            batch_id: the ID of the batch to retrieve.

        Returns:
            The batch retrieved from the OpenAI Batch API.

        Raises:
            openai.OpenAIError: if there was an error while retrieving the batch from the
                OpenAI Batch API.
        """
        import openai

        try:
            return self._client.batches.retrieve(batch_id)
        except openai.OpenAIError as e:
            self._logger.error(  # type: ignore
                f"Error while retrieving batch '{batch_id}' from OpenAI: {e}"
            )
            raise e

    def _retrieve_batch_results(self, batch: "OpenAIBatch") -> List[Dict[str, Any]]:
        """Retrieves the results of a batch from its output file, parsing the JSONL content
        into a list of dictionaries.

        Args:
            batch: the batch to retrieve the results from.

        Returns:
            A list of dictionaries containing the results of the batch.

        Raises:
            AssertionError: if no output file ID was found in the batch.
        """
        import openai

        assert batch.output_file_id, "No output file ID was found in the batch."

        try:
            file_response = self._client.files.content(batch.output_file_id)
            return [orjson.loads(line) for line in file_response.text.splitlines()]
        except openai.OpenAIError as e:
            self._logger.error(  # type: ignore
                f"Error while retrieving batch results from file '{batch.output_file_id}': {e}"
            )
            return []

    def _create_jobs(
        self, inputs: List["FormattedInput"], **kwargs: Any
    ) -> Tuple[str, ...]:
        """Creates jobs in the OpenAI Batch API to generate responses for the given inputs.

        Args:
            inputs: a list of inputs in chat format to generate responses for.
            kwargs: the keyword arguments to use for the generation.

        Returns:
            A list of job IDs created in the OpenAI Batch API.
        """
        batch_input_files = self._create_batch_files(inputs=inputs, **kwargs)
        jobs = []
        for batch_input_file in batch_input_files:
            if batch := self._create_batch_api_job(batch_input_file):
                jobs.append(batch.id)
        return tuple(jobs)

    def _create_batch_api_job(
        self, batch_input_file: "OpenAIFileObject"
    ) -> Union["OpenAIBatch", None]:
        """Creates a job in the OpenAI Batch API to generate responses for the given input
        file.

        Args:
            batch_input_file: the input file to generate responses for.

        Returns:
            The batch job created in the OpenAI Batch API.
        """
        import openai

        metadata = {"description": "distilabel"}

        if distilabel_pipeline_name := envs.DISTILABEL_PIPELINE_NAME:
            metadata["distilabel_pipeline_name"] = distilabel_pipeline_name

        if distilabel_pipeline_cache_id := envs.DISTILABEL_PIPELINE_CACHE_ID:
            metadata["distilabel_pipeline_cache_id"] = distilabel_pipeline_cache_id

        batch = None
        try:
            batch = self._client.batches.create(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=batch_input_file.id,
                metadata=metadata,
            )
        except openai.OpenAIError as e:
            self._logger.error(  # type: ignore
                f"Error while creating OpenAI Batch API job for file with ID"
                f" '{batch_input_file.id}': {e}."
            )
            raise e
        return batch

    def _create_batch_files(
        self, inputs: List["FormattedInput"], **kwargs: Any
    ) -> List["OpenAIFileObject"]:
        """Creates the necessary input files for the batch API to generate responses. The
        maximum size of each file so the OpenAI Batch API can process it is 100MB, so we
        need to split the inputs into multiple files if necessary.

        More information: https://platform.openai.com/docs/api-reference/files/create

        Args:
            inputs: a list of inputs in chat format to generate responses for, optionally
                including structured output.
            kwargs: the keyword arguments to use for the generation.

        Returns:
            The list of file objects created for the OpenAI Batch API.

        Raises:
            openai.OpenAIError: if there was an error while creating the batch input file
                in the OpenAI Batch API.
        """
        import openai

        files = []
        for file_no, buffer in enumerate(
            self._create_jsonl_buffers(inputs=inputs, **kwargs)
        ):
            try:
                # TODO: add distilabel pipeline name and id
                batch_input_file = self._client.files.create(
                    file=(self._name_for_openai_files(file_no), buffer),
                    purpose="batch",
                )
                files.append(batch_input_file)
            except openai.OpenAIError as e:
                self._logger.error(  # type: ignore
                    f"Error while creating OpenAI batch input file: {e}"
                )
                raise e
        return files

    def _create_jsonl_buffers(
        self, inputs: List["FormattedInput"], **kwargs: Any
    ) -> Generator[io.BytesIO, None, None]:
        """Creates a generator of buffers containing the JSONL formatted inputs to be
        used by the OpenAI Batch API. The buffers created are of size 100MB or less.

        Args:
            inputs: a list of inputs in chat format to generate responses for, optionally
                including structured output.
            kwargs: the keyword arguments to use for the generation.

        Yields:
            A buffer containing the JSONL formatted inputs to be used by the OpenAI Batch
            API.
        """
        buffer = io.BytesIO()
        buffer_current_size = 0
        for i, input in enumerate(inputs):
            # We create the smallest `custom_id` so we don't  increase the size of the file
            # to much, but we can still sort the results with the order of the inputs.
            row = self._create_jsonl_row(input=input, custom_id=str(i), **kwargs)
            row_size = len(row)
            if row_size + buffer_current_size > _OPENAI_BATCH_API_MAX_FILE_SIZE:
                buffer.seek(0)
                yield buffer
                buffer = io.BytesIO()
                buffer_current_size = 0
            buffer.write(row)
            buffer_current_size += row_size

        if buffer_current_size > 0:
            buffer.seek(0)
            yield buffer

    def _create_jsonl_row(
        self, input: "FormattedInput", custom_id: str, **kwargs: Any
    ) -> bytes:
        """Creates a JSONL formatted row to be used by the OpenAI Batch API.

        Args:
            input: a list of inputs in chat format to generate responses for, optionally
                including structured output.
            custom_id: a custom ID to use for the row.
            kwargs: the keyword arguments to use for the generation.

        Returns:
            A JSONL formatted row to be used by the OpenAI Batch API.
        """
        # TODO: depending on the format of the input, add `response_format` to the kwargs
        row = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"messages": input, **kwargs},
        }
        json_row = orjson.dumps(row)
        return json_row + b"\n"

    def _name_for_openai_files(self, file_no: int) -> str:
        if (
            envs.DISTILABEL_PIPELINE_NAME is None
            or envs.DISTILABEL_PIPELINE_CACHE_ID is None
        ):
            return f"distilabel-pipeline-fileno-{file_no}.jsonl"

        return f"distilabel-pipeline-{envs.DISTILABEL_PIPELINE_NAME}-{envs.DISTILABEL_PIPELINE_CACHE_ID}-fileno-{file_no}.jsonl"

    @staticmethod
    def _get_llm_statistics(
        completion: Union["OpenAIChatCompletion", "OpenAICompletion"],
    ) -> "LLMStatistics":
        return {
            "output_tokens": [
                completion.usage.completion_tokens if completion.usage else 0
            ],
            "input_tokens": [completion.usage.prompt_tokens if completion.usage else 0],
        }
