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

from __future__ import annotations

import multiprocessing as mp
import queue
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Type,
    Union,
)

from distilabel.logger import get_logger
from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from distilabel.llm.utils import LLMOutput
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class LLM(ABC):
    def __init__(
        self,
        task: Task,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the LLM base class.

        Note:
            This class is intended to be used internally, but you anyone can still create
            a subclass, implement the `abstractmethod`s and use it.

        Args:
            task (Task): the task to be performed by the LLM.
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
                Defaults to `None`.
            prompt_format (Union["SupportedFormats", None], optional): the format to be used
                for the prompt. If `None`, the default format of the task will be used, available
                formats are `openai`, `chatml`, `llama2`, `zephyr`, and `default`. Defaults to `None`,
                but `default` (concatenation of `system_prompt` and `formatted_prompt` with a line-break)
                will be used if no `prompt_formatting_fn` is provided.
            prompt_formatting_fn (Union[Callable[..., str], None], optional): a function to be
                applied to the prompt before generation. If `None`, no formatting will be applied.
                Defaults to `None`.
        """
        self.task = task

        self.thread_pool_executor = (
            ThreadPoolExecutor(max_workers=num_threads)
            if num_threads is not None
            else None
        )

        self.prompt_format = prompt_format
        self.prompt_formatting_fn = prompt_formatting_fn

    def __del__(self) -> None:
        """Shuts down the thread pool executor if it is not `None`."""
        if self.thread_pool_executor is not None:
            self.thread_pool_executor.shutdown()

    @property
    def num_threads(self) -> Union[int, None]:
        if self.thread_pool_executor:
            return self.thread_pool_executor._max_workers

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task.__class__.__name__}, num_threads={self.num_threads}, promp_format='{self.prompt_format}', model='{self.model_name}')"

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield "task", self.task
        yield "num_threads", self.num_threads
        yield "prompt_format", self.prompt_format
        if self.prompt_formatting_fn is not None:
            args = f"({', '.join(self.prompt_formatting_fn.__code__.co_varnames)})"
            representation = self.prompt_formatting_fn.__name__ + args
            yield "prompt_formatting_fn", representation
        yield "model", self.model_name

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def _generate_prompts(
        self,
        inputs: List[Dict[str, Any]],
        default_format: Union["SupportedFormats", None] = None,
        expected_output_type: Type = str,
    ) -> List[Any]:
        """Generates the prompts to be used for generation.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            default_format (Union["SupportedFormats", None], optional): the default format to be used
                for the prompt if no `prompt_format` is specified. Defaults to `None`.
            expected_output_type (Type, optional): the expected type of the prompt. Defaults to `str`.

        Returns:
            List[Any]: the generated prompts.

        Raises:
            ValueError: if the generated prompt is not of the expected type.
        """
        prompts = []
        for input in inputs:
            prompt = self.task.generate_prompt(**input)
            if not isinstance(prompt, Prompt) and self.prompt_formatting_fn is not None:
                warnings.warn(
                    "The method `generate_prompt` is not returning a `Prompt` class but a prompt"
                    f" of `type={type(prompt)}`, meaning that a pre-formatting has already been"
                    " applied in the `task.generate_prompt` method, so the usage of a `prompt_formatting_fn`"
                    " is discouraged.",
                    UserWarning,
                    stacklevel=2,
                )
                prompt = self.prompt_formatting_fn(prompt)
            elif isinstance(prompt, Prompt) and self.prompt_formatting_fn is None:
                if self.prompt_format is not None or default_format is not None:
                    prompt = prompt.format_as(
                        format=self.prompt_format or default_format  # type: ignore
                    )
                else:
                    warnings.warn(
                        "No `prompt_format` has been specified and no `default_format` is set, so"
                        " the prompt will be concatenated with a line-break and no specific formatting"
                        " by default.",
                        UserWarning,
                        stacklevel=2,
                    )
                    prompt = prompt.format_as(format="default")
            if not isinstance(prompt, expected_output_type):
                raise ValueError(
                    f"The provided `prompt={prompt}` is of `type={type(prompt)}`, but it must be of"
                    f" `type={expected_output_type}`, so make sure that `task.generate_prompt` returns"
                    f" a `{expected_output_type}` or that the `formatting_fn` formats the prompt as a "
                    f" `{expected_output_type}`."
                )
            prompts.append(prompt)
        return prompts

    @abstractmethod
    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        pass

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> Union[List[Future[List["LLMOutput"]]], List[List["LLMOutput"]]]:
        """Generates the outputs for the given inputs using the LLM.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each input.
                Defaults to `1`.
            progress_callback_func (Union[Callable, None], optional): a function to be called at each
                generation step. Defaults to `None`.

        Returns:
            Union[List[Future[List["LLMOutput"]]], List[List["LLMOutput"]]]: the generated outputs.
        """

        def _progress():
            if progress_callback_func is not None:
                advance = (
                    num_generations * len(inputs)
                    if not self.return_futures
                    else num_generations
                )
                progress_callback_func(advance=advance)

        if self.thread_pool_executor is not None:
            futures = []
            for input in inputs:
                future = self.thread_pool_executor.submit(
                    self._generate, [input], num_generations
                )
                future.add_done_callback(lambda _: _progress())
                futures.append(future)
            return futures

        generations = self._generate(inputs, num_generations)
        _progress()
        return generations

    @property
    def return_futures(self) -> bool:
        """Returns whether the LLM returns futures or not."""
        return self.thread_pool_executor is not None


class _TextGenerationRequest:
    def __init__(self, inputs, num_generations) -> None:
        self.future = Future()
        self.inputs = inputs
        self.num_generations = num_generations


class _TextGenerationCall:
    def __init__(self, inputs, num_generations) -> None:
        self.inputs = inputs
        self.num_generations = num_generations


class _TextGenerationResult:
    def __init__(self, generations) -> None:
        self.generations = generations


class _GenerationProcess(mp.Process):
    def __init__(self, process_llm: "ProcessLLM") -> None:
        self._task = process_llm.task
        # The function that will be executed to load the `LLM`
        self._load_llm_fn = process_llm._load_llm_fn

        # The `Semaphore` that will be used to synchronize the loading of the `LLM`.
        self._load_llm_sem = process_llm._load_llm_sem

        # Communication queues
        self._call_queue = process_llm._call_queue
        self._result_queue = process_llm._result_queue

        super().__init__()

    def _load_llm(self) -> LLM:
        llm = self._load_llm_fn(self._task)
        self._load_llm_sem.release()
        return llm

    def run(self) -> None:
        llm = self._load_llm()

        while True:
            request = self._call_queue.get()
            logger.info(f"Received request: {request}")
            if request is None:
                break
            generations = llm.generate(
                inputs=request.inputs, num_generations=request.num_generations
            )
            self._result_queue.put(_TextGenerationResult(generations))

    def stop(self) -> None:
        self._call_queue.put(None)


class _BridgeThread(Thread):
    def __init__(self, process_llm: "ProcessLLM") -> None:
        # The `Semaphore` that will be used to synchronize the loading of the `LLM`.
        self._load_llm_sem = process_llm._load_llm_sem

        # Communication queues between the main process and the child process
        self._call_queue = process_llm._call_queue
        self._result_queue = process_llm._result_queue

        self._pending_text_generation_request = (
            process_llm.pending_text_generation_request
        )
        self._text_generation_request_ids_queue = (
            process_llm.text_generation_request_ids_queue
        )

        super().__init__()

    def _wait_llm_loaded(self) -> None:
        logger.info("Waiting for the LLM to be loaded...")
        self._load_llm_sem.acquire()
        logger.info("LLM loaded!")

    def _get_text_generation_request(self) -> _TextGenerationRequest:
        text_generation_request_id = self._text_generation_request_ids_queue.get()
        return self._pending_text_generation_request[text_generation_request_id]

    def _call_generation_process(
        self, text_generation_request: _TextGenerationRequest
    ) -> None:
        text_generation_call = _TextGenerationCall(
            inputs=text_generation_request.inputs,
            num_generations=text_generation_request.num_generations,
        )
        self._call_queue.put(text_generation_call)

    def _get_result_generation_process(self) -> _TextGenerationResult:
        return self._result_queue.get()

    def _process_request(self) -> bool:
        # Get a text generation request
        text_generation_request_id = self._text_generation_request_ids_queue.get()
        if text_generation_request_id is None:
            return True

        tg_request = self._pending_text_generation_request[text_generation_request_id]
        tg_request.future.set_running_or_notify_cancel()

        # Send the text generation request to the child process
        self._call_generation_process(tg_request)

        # Get the text generation result from the child process
        generation_result = self._get_result_generation_process()

        # Set the result of the text generation request
        tg_request.future.set_result(generation_result.generations)

        return False

    def run(self) -> None:
        self._wait_llm_loaded()
        while True:
            should_stop = self._process_request()
            if should_stop:
                break

    def stop(self) -> None:
        self._text_generation_request_ids_queue.put(None)


class ProcessLLM:
    """A class that wraps an `LLM` and performs generation in a separate process."""

    def __init__(self, task: Task, load_llm_fn: Callable[..., LLM]) -> None:
        """Initializes the `ProcessLLM` class.

        Args:
            load_llm_fn (Callable[..., LLM]): a function that will be executed in the
                child process to load the `LLM`. It must return an `LLM` instance.
        """
        self.task = task

        self._load_llm_fn = load_llm_fn

        # The bridge thread will act as a bridge between the main process and the child
        # process for communication. It will send the generation requests to the child
        # process and receive the results from the child process.
        self._bridge_thread = None

        # The child process which will load the `LLM` and perform the generation.
        self._generation_process = None

        # The `Semaphore` that will be used to synchronize the loading of the `LLM`.
        # `_BridgeThread` will be blocked until `_GenerationProcess` has called the
        # `load_llm_fn` and the `LLM` has been loaded.
        self._load_llm_sem = mp.Semaphore(0)

        # This thread will create text generation requests
        self.pending_text_generation_request: Dict[int, _TextGenerationRequest] = {}
        self.text_generation_request_count = 0
        self.text_generation_request_ids_queue = queue.Queue[int]()

        # Queues for the communication between the `_BridgeThread` and the `_GenerationProcess`
        self._call_queue = mp.Queue()
        self._result_queue = mp.Queue()

    def _start_bridge_thread(self) -> None:
        if self._bridge_thread is None:
            self._bridge_thread = _BridgeThread(self)
            self._bridge_thread.start()
            self._generation_process = _GenerationProcess(self)
            self._generation_process.start()

    def _add_text_generation_request(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> Future:
        text_generation_request = _TextGenerationRequest(
            inputs=inputs, num_generations=num_generations
        )
        self.pending_text_generation_request[
            self.text_generation_request_count
        ] = text_generation_request
        self.text_generation_request_ids_queue.put(self.text_generation_request_count)
        self.text_generation_request_count += 1
        return text_generation_request.future

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> Future[List["LLMOutput"]]:
        self._start_bridge_thread()
        return self._add_text_generation_request(inputs, num_generations)

    def shutdown(self) -> None:
        self._generation_process.stop()
        self._generation_process.join()

        self._bridge_thread.stop()
        self._bridge_thread.join()

    @property
    def return_futures(self) -> bool:
        return True
