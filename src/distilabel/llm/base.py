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

import queue
import random
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from ctypes import c_char
from functools import cached_property
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

import multiprocess as mp

from distilabel.logger import get_logger
from distilabel.tasks.prompt import Prompt
from distilabel.utils.futures import when_all_complete

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
    ) -> Union[List[List["LLMOutput"]], Future[List[List["LLMOutput"]]]]:
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
                progress_callback_func(advance=num_generations * len(inputs))

        if self.thread_pool_executor is not None:
            futures = []
            for input in inputs:
                future = self.thread_pool_executor.submit(
                    self._generate, [input], num_generations
                )
                futures.append(future)
            future = when_all_complete(futures)
            future.add_done_callback(lambda _: _progress())
            return future

        generations = self._generate(inputs, num_generations)
        _progress()
        return generations

    @property
    def return_futures(self) -> bool:
        """Whether the `LLM` returns futures"""
        return True


MAX_MODEL_NAME_LENGTH = 256


class _TextGenerationRequest:
    """An object used to transfer the text generation request from the main process to
    the `_BridgeThread`."""

    def __init__(self, inputs: List[Dict[str, Any]], num_generations: int) -> None:
        self.future = Future()
        self.inputs = inputs
        self.num_generations = num_generations


class _TextGenerationCall:
    """An object used to transfer the text generation info from the `_BridgeThread` to
    the `_GenerationProcess`."""

    def __init__(self, inputs: List[Dict[str, Any]], num_generations: int) -> None:
        self.inputs = inputs
        self.num_generations = num_generations


class _TextGenerationResult:
    """An object used to transfer the text generation results from the `_GenerationProcess`
    to the `_BridgeThread`."""

    def __init__(
        self,
        generations: Union[List[List["LLMOutput"]], None] = None,
        exception: Union[Exception, None] = None,
    ) -> None:
        self.generations = generations
        self.exception = exception


class _GenerationProcess(mp.Process):
    """A process that will load the `LLM` and perform the text generation.

    This process will load the `LLM` using the `load_llm_fn` and then it will wait to
    receive a `_TextGenerationCall` from the `_BridgeThread`. Once it receives the
    `_TextGenerationCall`, it will perform the text generation and send the
    `_TextGenerationResult` to the `_BridgeThread`.
    """

    def __init__(self, process_llm: "ProcessLLM") -> None:
        self._task = process_llm.task
        # The function that will be executed to load the `LLM`
        self._load_llm_fn = process_llm._load_llm_fn

        # The `Semaphore` that will be used to synchronize the loading of the `LLM`.
        self._load_llm_sem = process_llm._load_llm_sem

        # Communication queues
        self._call_queue = process_llm._call_queue
        self._result_queue = process_llm._result_queue

        # Shared memory object for transfering the `model_name` to the main process
        self._model_name = process_llm._model_name

        super().__init__(daemon=True)

    def _load_llm(self) -> LLM:
        """Loads the `LLM` and sets the model name in the shared memory object."""
        llm = self._load_llm_fn(self._task)
        self._set_model_name(llm.model_name)
        logger.debug(
            f"Loaded '{llm.__class__.__name__}' with model '{llm.model_name}'!"
        )
        self._load_llm_sem.release()
        return llm

    def _set_model_name(self, model_name: str) -> None:
        """Sets the model name in the shared memory object, so the main process can
        access it."""
        truncated_model_name = model_name[: MAX_MODEL_NAME_LENGTH - 1].encode("utf-8")
        with self._model_name:
            for i, c in enumerate(truncated_model_name):
                self._model_name[i] = c
            self._model_name[len(truncated_model_name)] = b"\0"

    def run(self) -> None:
        """Runs the infinite loop of the generation process. It will wait for a text
        generation request from the bridge thread, perform the generation and send the
        result to the bridge thread.
        """
        llm = self._load_llm()
        name = f"{llm.__class__.__name__}({llm.model_name})"

        while True:
            request = self._call_queue.get()
            if request == -1:
                logger.debug(
                    f"Process with '{name}' received stop request. Stopping generation process..."
                )
                break

            # Perform generation
            logger.debug(f"Process with '{name}' received request...")
            try:
                generations = llm.generate(
                    inputs=request.inputs, num_generations=request.num_generations
                )
            except Exception as e:
                logger.error(
                    f"Process with '{name}' failed to perform generation with error: {e}"
                )
                generations = e

            if isinstance(generations, Exception):
                text_generation_result = _TextGenerationResult(exception=generations)
            elif isinstance(generations, Future):
                text_generation_result = _TextGenerationResult(
                    generations=generations.result()
                )
            else:
                text_generation_result = _TextGenerationResult(generations=generations)

            self._result_queue.put(text_generation_result)

    def stop(self) -> None:
        """Stops the infinite loop of the generation process."""
        self._call_queue.put(-1)


class _BridgeThread(Thread):
    """A thread that will act as a bridge between the main process and the child process.

    It will receive the text generation requests from the main process and send them to
    the child process. The main process will get a `Future` associated to the request.
    In order to communicate with the generation process, the bridge thread will use
    `multiprocessing.Queue`s. For each `_TextGenerationRequest`, it will create a
    `_TextGenerationCall` and send it to the child process using the `_call_queue`, then
    the thread will get blocked until it receives the `_TextGenerationResult` from the
    child process using the `_result_queue`. Once the result is received, the thread will
    set the result of the `Future` associated to the request.
    """

    def __init__(self, process_llm: "ProcessLLM") -> None:
        # The `Semaphore` that will be used to synchronize the loading of the `LLM`.
        self._load_llm_sem = process_llm._load_llm_sem

        self._generation_process = process_llm._generation_process

        # Communication queues between the main process and the child process
        self._call_queue = process_llm._call_queue
        self._result_queue = process_llm._result_queue

        # Pending text generation requests and queue for communication between
        # the main thread and the `_BridgeThread`
        self._pending_text_generation_request = (
            process_llm.pending_text_generation_request
        )
        self._text_generation_request_ids_queue = (
            process_llm.text_generation_request_ids_queue
        )

        self._model_name = process_llm._model_name

        super().__init__(daemon=True)

    def _wait_llm_loaded(self) -> None:
        """Waits for the generation process to load the `LLM`."""
        generation_process_pid = self._generation_process.pid
        logger.debug(
            f"Waiting for process with PID {generation_process_pid} to load the LLM..."
        )
        self._load_llm_sem.acquire()
        logger.debug(f"Process with PID {generation_process_pid} has loaded the LLM!")

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
        """Processes a text generation request. Returns `True` if the bridge thread
        should stop, `False` otherwise.

        Returns:
            bool: `True` if the bridge thread should stop, `False` otherwise.
        """
        # Get a text generation request
        text_generation_request_id = self._text_generation_request_ids_queue.get()
        if text_generation_request_id == -1:
            return True

        tg_request = self._pending_text_generation_request[text_generation_request_id]
        tg_request.future.set_running_or_notify_cancel()

        # Send the text generation request to the child process
        self._call_generation_process(tg_request)

        # Get the text generation result from the child process
        logger.debug(
            f"Bridge thread waiting for generation result with request id {text_generation_request_id}..."
        )
        generation_result = self._result_queue.get()
        if generation_result == -1:
            return True

        if generation_result.exception is not None:
            # Set the exception of the text generation request
            tg_request.future.set_exception(generation_result.exception)
        else:
            # Set the result of the text generation request
            tg_request.future.set_result(generation_result.generations)

        return False

    def run(self) -> None:
        """Runs the infinite loop of the bridge thread. It will wait for a text generation
        request from the main process, send it to the child process, wait for the result
        and set the result of the `Future` associated to the request.
        """

        self._wait_llm_loaded()
        while True:
            should_stop = self._process_request()
            if should_stop:
                break

        logger.debug("Bridge thread stopped!")

    def stop(self) -> None:
        """Stops the infinite loop of the bridge thread."""
        self._text_generation_request_ids_queue.put(-1)
        # This is for making sure that if the bridge thread has sent a request to the
        # generation process, and the generation process is stopped before sending the
        # result, the bridge thread will not get blocked waiting for the result.
        self._result_queue.put(-1)


class ProcessLLM:
    """A class that wraps an `LLM` and performs generation in a separate process. The
    result is a `Future` that will be set when the generation is completed.

    This class creates a new child process that will load the `LLM` and perform the
    text generation. In order to communicate with this child process, a bridge thread
    is created in the main process. The bridge thread will send and receive the results
    from the child process using `multiprocessing.Queue`s. The communication between the
    bridge thread and the main process is done using `Future`s. This architecture was
    inspired by the `ProcessPoolExecutor` from the `concurrent.futures` module and it's
    a simplified version of it.
    """

    def __init__(self, task: Task, load_llm_fn: Callable[[Task], LLM]) -> None:
        """Initializes the `ProcessLLM` class.

        Args:
            task: the task to be performed by the `LLM`. This task will be used by the
                child process when calling the `load_llm_fn`.
            load_llm_fn (Callable[[Task], LLM]): a function that will be executed in the
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
        self.text_generation_request_ids_queue: queue.Queue[int] = queue.Queue()

        # Queues for the communication between the `_BridgeThread` and the `_GenerationProcess`
        self._call_queue = mp.Queue()
        self._result_queue = mp.Queue()

        # Shared memory object for transfering the `model_name` to the main process
        # once the `LLM` is loaded
        self._model_name = mp.Array(c_char, MAX_MODEL_NAME_LENGTH)

    def _start_bridge_thread(self) -> None:
        """Starts the bridge thread and the generation process."""
        if self._bridge_thread is None:
            self._generation_process = _GenerationProcess(self)
            self._generation_process.start()
            pid = self._generation_process.pid
            logger.debug(f"Generation process with PID {pid} started!")

            self._bridge_thread = _BridgeThread(self)
            self._bridge_thread.start()
            logger.debug("Bridge thread for process with PID {pid} started!")

    def _add_text_generation_request(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> Future[List[List["LLMOutput"]]]:
        """Creates and send a new text generation request to the bridge thread. This thread
        and the bridge thread shares a dictionary used to store the text generation requests.
        This thread will add the text generation requests to the dictionary and the bridge
        thread will only read from it. In order for the bridge thread to know that a new
        text generation request has been added to the dictionary, this thread will put the
        id of the request in a queue. The bridge thread will read from this queue and get
        the text generation request from the dictionary.
        """

        def _progress():
            if progress_callback_func is not None:
                progress_callback_func(advance=num_generations * len(inputs))

        text_generation_request = _TextGenerationRequest(
            inputs=inputs, num_generations=num_generations
        )
        # Put the request information in the dictionary associated to the request id
        self.pending_text_generation_request[
            self.text_generation_request_count
        ] = text_generation_request
        # Put the request id in the queue (for the `_BridgeThread` to consume it)
        self.text_generation_request_ids_queue.put(self.text_generation_request_count)
        self.text_generation_request_count += 1
        text_generation_request.future.add_done_callback(lambda _: _progress())
        return text_generation_request.future

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> Future[List[List["LLMOutput"]]]:
        """Generates the outputs for the given inputs using the `ProcessLLM` and its loaded
        `LLM`.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each input.
                Defaults to `1`.
            progress_callback_func (Union[Callable, None], optional): a function to be called at each
                generation step. Defaults to `None`.

        Returns:
            Future[List[List["LLMOutput"]]]: the generated outputs as a `Future`.
        """
        self._start_bridge_thread()
        return self._add_text_generation_request(
            inputs, num_generations, progress_callback_func
        )

    def teardown(self) -> None:
        """Stops the bridge thread and the generation process."""
        if self._generation_process is not None:
            self._generation_process.stop()
            self._generation_process.join()

        if self._bridge_thread is not None:
            self._bridge_thread.stop()
            self._bridge_thread.join()

    @cached_property
    def model_name(self) -> str:
        """Returns the model name of the `LLM` once it has been loaded."""
        with self._model_name:
            return "".join([c.decode() for c in self._model_name if c != b"\0"])

    @property
    def return_futures(self) -> bool:
        """Whether the `LLM` returns futures"""
        return True


class LLMPool:
    """LLMPool is a class that wraps multiple `ProcessLLM`s and performs generation in
    parallel using them. Depending on the number of `LLM`s and the parameter `num_generations`,
    the `LLMPool` will decide how many generations to perform for each `LLM`:

    - If `num_generations` is less than the number of `LLM`s, then `num_generations` LLMs
    will be chosen randomly and each of them will perform 1 generation.


    - If `num_generations` is equal to the number of `LLM`s, then each `LLM` will perform
    1 generation.

    - If `num_generations` is greater than the number of `LLM`s, then each `LLM` will
    perform `num_generations // num_llms` generations, and the remaining `num_generations % num_llms`
    generations will be performed by `num_generations % num_llms` randomly chosen `LLM`s.

    Attributes:
        llms (List[ProcessLLM]): the `ProcessLLM`s to be used for generation.
    """

    def __init__(self, llms: List[ProcessLLM]) -> None:
        """Initializes the `LLMPool` class.

        Args:
            llms: the `ProcessLLM`s to be used for generation. The list must contain at
                least 2 `ProcessLLM`s.

        Raises:
            ValueError: if the `llms` argument contains less than 2 `ProcessLLM`s, the
                `llms` argument contains `ProcessLLM`s that are not `ProcessLLM`s, or
                if the `llms` argument contains `ProcessLLM`s with different tasks.
        """
        if len(llms) < 2:
            raise ValueError(
                "The `llms` argument must contain at least 2 `ProcessLLM`s. If you want"
                " to use a single `ProcessLLM`, use the `ProcessLLM` directly instead."
            )

        if not all(isinstance(llm, ProcessLLM) for llm in llms):
            raise ValueError("The `llms` argument must contain only `ProcessLLM`s.")

        if not all(llm.task == llms[0].task for llm in llms):
            raise ValueError(
                "The `llms` argument must contain `ProcessLLM`s with the same task."
            )

        self.llms = llms
        self.num_llms = len(llms)

    def _get_num_generations_per_llm(self, num_generations: int) -> Dict[int, int]:
        """Returns the number of generations to be performed by each `LLM`.

        Args:
            num_generations: the number of generations to be performed.

        Returns:
            Dict[int, int]: a dictionary where the keys are the ids of the `LLM`s and the
            values are the number of generations to be performed by each `LLM`.
        """
        llms_ids = list(range(self.num_llms))
        generations_per_llm = {i: num_generations // self.num_llms for i in llms_ids}

        for i in random.sample(llms_ids, k=num_generations % self.num_llms):
            generations_per_llm[i] += 1

        return generations_per_llm

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> List[List["LLMOutput"]]:
        """Generates the outputs for the given inputs using the pool of `ProcessLLM`s.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each input.
                Defaults to `1`.
            progress_callback_func (Union[Callable, None], optional): a function to be called at each
                generation step. Defaults to `None`.

        Returns:
            Future[List[List["LLMOutput"]]]: the generated outputs as a `Future`.
        """
        num_generations_per_llm = self._get_num_generations_per_llm(num_generations)

        futures = [
            llm.generate(
                inputs,
                num_generations=num_generations_per_llm[i],
                progress_callback_func=progress_callback_func,
            )
            for i, llm in enumerate(self.llms)
            if num_generations_per_llm[i] > 0
        ]
        llms_generations = [future.result() for future in futures]

        generations = []
        for llms_row_generations in zip(*llms_generations):
            row_generations = []
            for llm_row_generations in llms_row_generations:
                for generation in llm_row_generations:
                    row_generations.append(generation)
            generations.append(row_generations)

        return generations

    def teardown(self) -> None:
        """Stops the `ProcessLLM`s."""
        for llm in self.llms:
            llm.teardown()

    @property
    def task(self) -> "Task":
        """Returns the task that will be used by the `ProcessLLM`s of this pool.

        Returns:
            Task: the task that will be used by the `ProcessLLM`s of this pool.
        """
        return self.llms[0].task

    @property
    def return_futures(self) -> bool:
        """Whether the `LLM` returns futures"""
        return False
