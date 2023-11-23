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

import math
import os
import warnings
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Union,
)

from datasets import Dataset, Split

from distilabel.dataset import CustomDataset
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.progress_bar import (
    _pipeline_progress,
    get_progress_bars_for_pipeline,
    use_progress_bar,
)
from distilabel.utils.dicts import combine_dicts

if TYPE_CHECKING:
    from distilabel.llm.base import LLM


logger = get_logger()


class Pipeline:
    def __init__(
        self,
        generator: Union["LLM", None] = None,
        labeller: Union["LLM", None] = None,
    ) -> None:
        """Initializes the Pipeline class.

        Args:
            generator (Union["LLM", None], optional): the LLM to be used for generation.
                Defaults to None.
            labeller (Union["LLM", None], optional): the LLM to be used for labelling.
                Defaults to None.

        Raises:
            ValueError: if no LLM is provided.

        Examples:
            >>> from distilabel.llm.huggingface import TransformersLLM
            >>> from distilabel.llm.openai_ import OpenAILLM
            >>> from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
            >>> from distilabel.tasks.text_generation.llama import Llama2TextGenerationTask
            >>> from distilabel.pipeline import Pipeline

            >>> generator = TransformersLLM(
            ...     model="meta-llama/Llama-2-7b-chat-hf",
            ...     tokenizer="meta-llama/Llama-2-7b-chat-hf",
            ...     task=Llama2TextGenerationTask(),
            ... )
            >>> labeller = OpenAILLM(
            ...     model="gpt-3.5-turbo",
            ...     task=UltraFeedbackTask.for_text_quality(),
            ... )
            >>> pipeline = Pipeline(generator=generator, labeller=labeller)
            >>> dataset = pipeline.generate(dataset=..., num_generations=1, batch_size=1)
        """
        self.generator = generator
        self.labeller = labeller

        if self.generator is None and self.labeller is None:
            raise ValueError("At least one LLM has to be provided to the pipeline")

    def __repr__(self) -> str:
        return (
            f"Pipeline(\n\tgenerator={self.generator},\n\tlabeller={self.labeller}\n)"
        )

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield "generator", self.generator
        yield "labeller", self.labeller

    def _validate_dataset(self, dataset: Dataset) -> None:
        """Validates that the provided dataset contains the columns needed by the LLMs, and
        warns the user if the columns to be generated already exist.

        Args:
            dataset (Dataset): the dataset to be validated.

        Raises:
            KeyError: if the dataset does not contain the columns needed by the LLMs.
        """
        # Generation LLM has not been provided, so the columns needed by the Labelling
        # LLM must be in the provided dataset
        if self.labeller is not None:
            if self.generator is None:
                try:
                    self.labeller.task.validate_dataset(dataset.column_names)
                except KeyError as err:
                    raise KeyError(
                        "Labelling LLM expects a dataset with at least the following"
                        f" columns: {self.labeller.task.input_args_names}, but the provided"
                        f" dataset just contains: {dataset.column_names}"
                    ) from err
            else:
                expected_columns = (
                    dataset.column_names + self.generator.task.output_args_names
                )
                try:
                    self.labeller.task.validate_dataset(expected_columns)
                except KeyError as err:
                    raise KeyError(
                        "Labelling LLM expects to receive the following columns after the"
                        f" generation process: {self.labeller.task.input_args_names}, but the"
                        f" provided dataset including the columns to generate just contains: {expected_columns}"
                    ) from err

        if self.generator is not None:
            try:
                self.generator.task.validate_dataset(dataset.column_names)
            except KeyError as err:
                raise KeyError(
                    "Generation LLM expects a dataset with the following columns:"
                    f" {self.generator.task.input_args_names}, but the provided dataset"
                    f" just contains: {dataset.column_names}"
                ) from err

        # Additionally, we need to check that if the columns to be generated already exist,
        # then we should look for `None`/`null` values and just fulfill those, while skipping
        # the rest. This is useful to be able to continue a generation that broke or a process
        # that was interrupted
        generated_columns = []
        if self.generator is not None:
            generated_columns += self.generator.task.output_args_names
        if self.labeller is not None:
            generated_columns += self.labeller.task.output_args_names

        if set(generated_columns) == set(dataset.column_names).intersection(
            set(generated_columns)
        ):
            warnings.warn(
                "The provided dataset already contains the columns to be generated:"
                f" {generated_columns}; which means that the generation process will"
                " be skipped for the rows with values for those columns. If you want"
                " to re-generate those columns, please remove them from the dataset.",
                UserWarning,
                stacklevel=2,
            )

    def _get_batch_generations(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int,
        progress_callback_func: Union[Callable, None] = None,
    ) -> List[Dict[str, Any]]:
        """Gets the batch generations for the given inputs, capturing the futures if the
        LLM returns them, and then processes the batch generations.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int): the number of generations to be performed for each
                input.
            progress_callback_func (Union[Callable, None], optional): the callback function
                to be called when the progress of the generation process changes. Defaults
                to None.

        Returns:
            List[Dict[str, Any]]: the processed batch generations.
        """
        batch_generations = self.generator.generate(
            inputs=inputs,
            num_generations=num_generations,
            progress_callback_func=progress_callback_func,
        )

        processed_generations = []
        if self.generator.return_futures:  # type: ignore
            for future in batch_generations:
                result = future.result()
                processed_generations.extend(result)
        else:
            processed_generations = batch_generations

        return self._process_batch_generations(batch_generations=processed_generations)

    def _process_batch_generations(
        self,
        batch_generations: List[List["LLMOutput"]],
    ) -> List[Dict[str, Any]]:
        """Processes the batch generations, combining the outputs of the LLMs into a single
        dictionary.

        Args:
            batch_generations (List[List["LLMOutput"]]): the batch generations to be processed.

        Returns:
            List[Dict[str, Any]]: the processed batch generations.
        """
        processed_generations = []
        for generations in batch_generations:
            processed_generation = {
                # Since all the generations for the same `model_name` also share the same
                # `prompt_used`, then we just keep the first element in `generations`
                "generation_model": generations[0]["model_name"],
                "generation_prompt": generations[0]["prompt_used"],
                "raw_generation_responses": [
                    generation["raw_output"] for generation in generations
                ],
            }
            try:
                processed_generation.update(
                    **combine_dicts(
                        *[generation["parsed_output"] for generation in generations]
                    )
                )
            except Exception as e:
                warnings.warn(
                    f"Generation processing step failed when combining dicts: {e}",
                    UserWarning,
                    stacklevel=2,
                )
            processed_generations.append(processed_generation)
        return processed_generations

    def _include_generator_outputs_as_inputs(
        self, inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Includes the outputs of the generator as inputs for the labeller.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for labelling.
            outputs (List[Dict[str, Any]]): the outputs of the generator.

        Returns:
            List[Dict[str, Any]]: the inputs to be used for labelling.
        """
        for input_, output in zip(inputs, outputs):
            # Skip the keys not required by the labelling LLM
            input_.update(
                {
                    k: v
                    for k, v in output.items()
                    if self.labeller is not None
                    and k in self.labeller.task.input_args_names
                }
            )
        return inputs

    def _process_batch_labels(
        self, batch_labels: List[List["LLMOutput"]]
    ) -> List[Dict[str, Any]]:
        """Processes the batch labels, combining the outputs of the LLMs into a single
        dictionary.

        Args:
            batch_labels (List[List["LLMOutput"]]): the batch labels to be processed.

        Returns:
            List[Dict[str, Any]]: the processed batch labels.
        """
        processed_labels = []
        for labels in batch_labels:
            for label in labels:
                if label["parsed_output"] is not None and not isinstance(
                    label["parsed_output"], (list, dict)
                ):
                    raise ValueError(
                        f"Unsupported type: {type(label['parsed_output'])}"
                    )

                processed_label = {
                    # Since all the generations for the same `model_name` also share the same
                    # `prompt_used`, then we just keep the first element in `generations`
                    "labelling_model": label["model_name"],
                    "labelling_prompt": label["prompt_used"],
                    "raw_labelling_response": label["raw_output"],
                }
                try:
                    if isinstance(label["parsed_output"], list):
                        processed_label.update(**combine_dicts(*label["parsed_output"]))
                    elif isinstance(label["parsed_output"], dict):
                        processed_label.update(**label["parsed_output"])
                except Exception as e:
                    warnings.warn(
                        f"Label processing step failed when combining dicts: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
                processed_labels.append(processed_label)
        return processed_labels

    def _transform_dataset_to_expected_format(
        self, rows: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Transforms the `datasets.Dataset` to the expected format required by the LLMs
        during the `generate` process.

        Args:
            rows (Dict[str, List[Any]]): the rows to be transformed.

        Returns:
            List[Dict[str, Any]]: the transformed rows.
        """
        length = len(next(iter(rows.values())))

        generator_column_names = []
        if self.generator is not None:
            generator_column_names = self.generator.task.input_args_names
        labeller_column_names = []
        if self.labeller is not None:
            labeller_column_names = self.labeller.task.input_args_names
        column_names = generator_column_names + labeller_column_names

        inputs = []
        for i in range(length):
            input = {
                col: values[i] for col, values in rows.items() if col in column_names
            }
            inputs.append(input)

        return inputs

    def _build_dataset(  # noqa: C901
        self,
        dataset: Dataset,
        generations: List[Dict[str, Any]],
        batch_labels: Union[List[Future["LLMOutput"]], List["LLMOutput"]],
    ) -> CustomDataset:
        """Builds the final dataset with either the generations, the labels, or both, depending
        on the LLMs provided to the `Pipeline`.

        Args:
            dataset (Dataset): the original dataset.
            generations (List[Dict[str, Any]]): the processed generations.
            batch_labels (Union[List[Future["LLMOutput"]], List["LLMOutput"]]): the processed
                batch labels.

        Returns:
            CustomDataset: the final dataset.

        Raises:
            RuntimeError: if the `Pipeline` fails during the generation or labelling steps.
        """
        if self.generator is None:
            generations = [{} for _ in range(len(dataset))]
        else:
            generator_column_names = [
                "generation_model",
                "generation_prompt",
                "raw_generation_responses",
            ] + self.generator.task.output_args_names

            if len(generations) < len(dataset):
                generations.extend(
                    [
                        {key: None for key in generator_column_names}
                        for _ in range(len(dataset) - len(generations))
                    ]
                )

            for generation in generations:
                for key in generator_column_names:
                    if key not in generation:
                        generation.update({key: None})

        if self.labeller is None:
            labels = [{} for _ in range(len(dataset))]
        else:
            # If the LLM returns futures, we need to wait for them to finish
            processed_labels = []
            if self.labeller.return_futures:
                for future in batch_labels:
                    try:
                        processed_labels.extend(future.result())
                    except Exception as e:
                        logger.error(
                            f"An error ocurred when getting the result from the labeller: {e}"
                        )
                        processed_labels.append(
                            [
                                LLMOutput(
                                    model_name=self.labeller.model_name,
                                    prompt_used=None,
                                    raw_output=None,
                                    parsed_output=None,
                                )
                            ]
                        )
            else:
                processed_labels = batch_labels
            labels = self._process_batch_labels(batch_labels=processed_labels)  # type: ignore

            labeller_column_names = [
                "labelling_model",
                "labelling_prompt",
                "raw_labelling_response",
            ] + self.labeller.task.output_args_names

            # Ensure the lengths of the labels and the dataset match (when pipeline
            # fails in an intermediate step, the labels may be shorter than the dataset)
            if len(labels) < len(dataset):
                labels.extend(
                    [
                        {key: None for key in labeller_column_names}
                        for _ in range(len(dataset) - len(labels))
                    ]
                )

            # Add missing keys/columns with a `None` value
            for label in labels:
                for key in labeller_column_names:
                    if key not in label:
                        label.update({key: None})

        _dataset = Dataset(
            arrow_table=dataset.flatten_indices().data, split=Split.TRAIN
        )
        _dataset = _dataset.map(lambda _: {**generations.pop(0), **labels.pop(0)})  # type: ignore
        # Dynamically remaps the `datasets.Dataset` to be a `CustomDataset` instance
        _dataset.__class__ = CustomDataset
        _dataset.task = self.labeller.task if self.labeller is not None else None  # type: ignore
        return _dataset  # type: ignore

    @use_progress_bar
    def generate(  # noqa: C901
        self,
        dataset: Dataset,
        num_generations: int = 1,
        batch_size: int = 1,
        enable_checkpoints: bool = True,
        display_progress_bar: bool = False,
        verbose: bool = True,
    ) -> CustomDataset:
        """Generates the outputs for the given dataset using the LLMs provided to the `Pipeline`.

        Args:
            dataset (Dataset): the dataset to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each
                input. Defaults to `1`.
            batch_size (int, optional): the batch size to be used for generation. Defaults to `1`.
            enable_checkpoints (bool, optional): whether to enable checkpoints or not. Defaults to `True`.
            display_progress_bar (bool, optional): whether to display the progress bar or not. Defaults to `False`.
            verbose (bool, optional): whether to display the logs or not. Defaults to `True`.

        Returns:
            CustomDataset: the final dataset.

        Raises:
            RuntimeError: if the `Pipeline` fails during the generation or labelling steps.
            UserWarning: if the `Pipeline` fails during the generation or labelling steps and
                `enable_checkpoints` is set to `False`.

        Examples:
            >>> from distilabel.llm.huggingface import TransformersLLM
            >>> from distilabel.llm.openai_ import OpenAILLM
            >>> from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
            >>> from distilabel.tasks.text_generation.llama import Llama2TextGenerationTask
            >>> from distilabel.pipeline import Pipeline

            >>> generator = TransformersLLM(
            ...     model="meta-llama/Llama-2-7b-chat-hf",
            ...     tokenizer="meta-llama/Llama-2-7b-chat-hf",
            ...     task=Llama2TextGenerationTask(),
            ... )
            >>> labeller = OpenAILLM(
            ...     model="gpt-3.5-turbo",
            ...     task=UltraFeedbackTask.for_text_quality(),
            ... )
            >>> pipeline = Pipeline(generator=generator, labeller=labeller)
            >>> dataset = pipeline.generate(dataset=..., num_generations=1, batch_size=1)
        """
        if not verbose:
            logger.setLevel("ERROR")

        if (
            self.labeller is not None
            and self.generator is not None
            and num_generations < 2
        ):
            warnings.warn(
                f"Provided `num_generations={num_generations}` which implies that the "
                "`generator` LLM will just run once, while the `labelling` LLM expects "
                "to recieve a list of N inputs to label, where N is > 1. If this is not "
                "intended, make sure to set `num_generations` to a value higher or "
                "equal to 2.",
                UserWarning,
                stacklevel=2,
            )

        self._validate_dataset(dataset)

        generations: List[Dict[str, Any]] = []
        batch_labels: Union[Future[List["LLMOutput"]], List["LLMOutput"]] = []

        (
            generation_progress_func,
            labelling_progress_func,
        ) = get_progress_bars_for_pipeline(
            num_rows=len(dataset),
            num_generations=num_generations,
            display_progress_bar=display_progress_bar,
        )

        num_batches = math.ceil(len(dataset) / batch_size)

        for batch_i, rows in enumerate(dataset.iter(batch_size=batch_size), start=1):
            logger.info(f"Processing batch {batch_i} of {num_batches}...")
            inputs = self._transform_dataset_to_expected_format(rows)  # type: ignore

            if self.generator is not None:
                logger.info(f"Calling generator for batch {batch_i}...")
                try:
                    batch_generations = self._get_batch_generations(
                        inputs, num_generations, generation_progress_func
                    )
                    generations.extend(batch_generations)
                except Exception as e:
                    if not enable_checkpoints:
                        raise RuntimeError(
                            "`Pipeline.generate` failed during generation step. Setting `enable_checkpoints=True` is recommended!"
                        ) from e
                    logger.error(
                        f"`Pipeline.generate` failed during generation step with exception: {e}"
                    )
                    return self._build_dataset(
                        dataset, generations=generations, batch_labels=batch_labels
                    )

                inputs = self._include_generator_outputs_as_inputs(
                    inputs, batch_generations
                )

            if self.labeller is not None:
                logger.info(f"Calling labeller for batch {batch_i}...")
                try:
                    # TODO: move to `self._get_batch_labels` (without awaiting futures)
                    batch_labels.extend(
                        self.labeller.generate(  # type: ignore
                            inputs=inputs,
                            # `num_generations` is always 1 because labelling the same input multiple times
                            # using the same LLM may not make sense
                            num_generations=1,
                            progress_callback_func=labelling_progress_func,
                        )
                    )
                except Exception as e:
                    if not enable_checkpoints:
                        raise RuntimeError(
                            "`Pipeline.generate` failed during labelling step. Setting `enable_checkpoints=True` is recommended!"
                        ) from e
                    logger.error(
                        f"`Pipeline.generate` failed during labelling step with exception: {e}"
                    )
                    return self._build_dataset(
                        dataset, generations=generations, batch_labels=batch_labels
                    )

        _pipeline_progress.stop()

        return self._build_dataset(
            dataset, generations=generations, batch_labels=batch_labels
        )


def pipeline(
    task: Literal["preference"],
    subtask: Optional[str] = None,
    *,
    generator: Optional["LLM"] = None,
    labeller: Optional["LLM"] = None,
    **kwargs,
) -> Pipeline:
    """Creates a `Pipeline` instance with the provided LLMs for a given task, which is useful
    whenever you want to use a pre-defined `Pipeline` for a given task, or if you want to
    create a custom `Pipeline` for a given task. Ideally one using this function over the `Pipeline`
    class, don't want to worry about the details of the `labeller`, since it will come with a default
    configuration based on the `task`, by default the LLM used for `labelling` will always be `gpt-3.5-turbo`
    from OpenAI, as it's the one that provides the most consistent and fast results.

    Args:
        task (Literal["preference", "critique"]): the task to be performed by the `Pipeline`.
        subtask (Optional[str], optional): the subtask to be performed by the `Pipeline`.
            Defaults to None.
        generator (Optional["LLM"], optional): the LLM to be used for generation. Defaults to None.
        labeller (Optional["LLM"], optional): the LLM to be used for labelling. Defaults to None.
        **kwargs: the keyword arguments to be passed to the `task` and `subtask` classes.

    Raises:
        ValueError: if an invalid task is provided.

    Returns:
        Pipeline: the `Pipeline` instance.

    Examples:
        >>> from distilabel.llm.huggingface import TransformersLLM
        >>> from distilabel.tasks.text_generation.llama import Llama2TextGenerationTask
        >>> from distilabel.pipeline import pipeline

        >>> generator = TransformersLLM(
        ...     model="meta-llama/Llama-2-7b-chat-hf",
        ...     tokenizer="meta-llama/Llama-2-7b-chat-hf",
        ...     task=Llama2TextGenerationTask(),
        ... )
        >>> pipeline = pipeline(
        ...     task="preference",
        ...     subtask="text-quality",
        ...     generator=generator,
        ... )
    """
    if task == "preference":
        if labeller is None:
            from dataclasses import fields

            from distilabel.llm.openai import OpenAILLM
            from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask

            task_cls = UltraFeedbackTask
            task_kwargs = {
                key: kwargs.get(key.name)
                for key in fields(task_cls)
                if key.name in kwargs and not key.name.startswith("__")
            }

            # Dynamically call the appropriate classmethod using getattr
            if subtask is not None:
                if subtask not in task_cls.__subtasks__:
                    raise ValueError(
                        f"Invalid subtask: {subtask}, available subtasks are {task_cls.__subtasks__}"
                    )
                classmethod_name = f"for_{subtask.lower().replace('-', '_')}"
                if hasattr(task_cls, classmethod_name):
                    task_cls = getattr(task_cls, classmethod_name)

            logger.info(
                "Since no `labeller` was provided, `OpenAILLM` will be used as the default labeller with `UltraFeedback`."
            )

            labeller = OpenAILLM(
                model=kwargs.get("openai_model") or "gpt-3.5-turbo",
                task=task_cls(**task_kwargs),  # type: ignore
                max_new_tokens=kwargs.get("max_new_tokens") or 256,
                num_threads=kwargs.get("num_threads") or 4,
                openai_api_key=kwargs.get("openai_api_key")
                or os.getenv("OPENAI_API_KEY"),
                temperature=kwargs.get("temperature") or 0.0,
            )
        else:
            from distilabel.tasks.preference.judgelm import JudgeLMTask
            from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
            from distilabel.tasks.preference.ultrajudge import UltraJudgeTask

            if not isinstance(
                labeller.task, (UltraFeedbackTask, JudgeLMTask, UltraJudgeTask)
            ):
                warnings.warn(
                    "The `labeller` task for `preference` must be an instance of `UltraFeedbackTask`,"
                    f" `JudgeLMTask` or `UltraJudge`, got {labeller.task.__class__.__name__}."
                    "If you are planning to use a custom `labeller` for a `preference` "
                    "task, use it at your own risk.",
                    UserWarning,
                    stacklevel=2,
                )

        if generator is not None:
            assert (
                generator.task.input_args_names + generator.task.output_args_names
                == labeller.task.input_args_names
            ), (
                f"`generator` outputs do not match `labeller` inputs: "
                f"{generator.task.input_args_names + generator.task.output_args_names} != {labeller.task.input_args_names}"
            )
    else:
        raise ValueError(f"Invalid task: {task}, available tasks are: `preference`.")

    return Pipeline(generator=generator, labeller=labeller)
