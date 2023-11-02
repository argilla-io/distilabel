import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

from datasets import Dataset

from ultralabel.dataset import CustomDataset
from ultralabel.progress_bar import get_progress_bars_for_pipeline
from ultralabel.utils import combine_dicts

if TYPE_CHECKING:
    from ultralabel.llm.base import LLM


T = TypeVar("T", bound=Dataset)


class Pipeline(Generic[T]):
    dataset_cls: Type[T] = CustomDataset

    def __init__(
        self,
        generator: Union["LLM", None] = None,
        labeller: Union["LLM", None] = None,
    ) -> None:
        self.generator = generator
        self.labeller = labeller

        if self.generator is None and self.labeller is None:
            raise ValueError("At least one LLM has to be provided to the pipeline")

    def _remap_dataset(self, dataset: Dataset) -> T:
        # Dynamically remaps the `datasets.Dataset` to be an instance of `dataset_cls`
        dataset.__class__ = self.dataset_cls
        return dataset  # type: ignore

    def _validate_dataset(self, dataset: Dataset) -> None:
        # Generation LLM has not been provided, so the columns needed by the Labelling
        # LLM must be in the provided dataset
        if self.labeller is not None:
            if self.generator is None:
                try:
                    self.labeller.task.validate_dataset(dataset.column_names)
                except KeyError as err:
                    raise KeyError(
                        f"Labelling LLM expects a dataset with the following columns: {self.labeller.task.input_args_names}"
                    ) from err
            else:
                expected_columns = (
                    dataset.column_names + self.generator.task.output_args_names
                )
                try:
                    self.labeller.task.validate_dataset(expected_columns)
                except KeyError as err:
                    raise KeyError(
                        f"Labelling LLM expects to receive the following columns after the generation process: {expected_columns}"
                    ) from err

        if self.generator is not None:
            try:
                self.generator.task.validate_dataset(dataset.column_names)
            except KeyError as err:
                raise KeyError(
                    f"Generation LLM expects a dataset with the following columns: {self.generator.task.input_args_names}"
                ) from err

    def _add_columns_to_dataset(
        self,
        dataset: Dataset,
        generations: List[Dict[str, Any]],
        labels: List[Dict[str, Any]],
    ) -> Dataset:
        if self.generator is not None:
            for output_name in self.generator.task.output_args_names:
                dataset = dataset.add_column(
                    output_name, [row.get(output_name, None) for row in generations]
                )

            dataset = dataset.add_column(
                "raw_generation_response",
                [row.get("raw_generation_response", None) for row in generations],
            )

        if self.labeller is not None:
            for output_name in self.labeller.task.output_args_names:
                dataset = dataset.add_column(
                    output_name, [row.get(output_name, None) for row in labels]
                )

            dataset = dataset.add_column(
                "raw_labelling_response",
                [row.get("raw_labelling_response", None) for row in labels],
            )

        return dataset

    def _get_batch_generations(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int,
        progress_callback_func: Union[Callable, None] = None,
    ) -> List[Dict[str, Any]]:
        batch_generations = self.generator.generate(
            inputs=inputs,
            num_generations=num_generations,
            progress_callback_func=progress_callback_func,
        )

        if self.generator.return_futures:
            batch_generations = [future.result() for future in batch_generations]

        # TODO(alvarobartt): implement fallback mechanism if `combine_dicts` fails
        return [
            {
                "raw_generation_response": raw_responses,
                **combine_dicts(*parsed_responses),
            }
            for raw_responses, parsed_responses in batch_generations
        ]

    def _transform_dataset_to_expected_format(
        self, rows: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        length = len(next(iter(rows.values())))

        inputs = []
        for i in range(length):
            input = {col: values[i] for col, values in rows.items()}
            inputs.append(input)

        return inputs

    def generate(  # noqa: C901
        self,
        dataset: Dataset,
        num_generations: int = 1,
        batch_size: int = 1,
        display_progress_bar: bool = False,
    ) -> T:
        # We need to convert the dataset to a dict and then back to a dataset
        # as we just want to keep the dataset content, not its metadata
        dataset = Dataset.from_dict(dataset.to_dict())
        self._validate_dataset(dataset)

        generations: List[Dict[str, Any]] = []
        labels: List[Dict[str, Any]] = []

        (
            generation_progress_func,
            labelling_progress_bar,
        ) = get_progress_bars_for_pipeline(
            num_rows=len(dataset),
            num_generations=num_generations,
            display_progress_bar=display_progress_bar,
        )

        for rows in dataset.iter(batch_size=batch_size):
            inputs = self._transform_dataset_to_expected_format(rows)

            if self.generator is not None:
                batch_generations = self._get_batch_generations(
                    inputs, num_generations, generation_progress_func
                )
                generations.extend(batch_generations)

                for input, generations_ in zip(inputs, batch_generations):
                    # Skip the `raw_generation_response` key as not used by the labelling LLM
                    input.update(
                        {
                            k: v
                            for k, v in generations_.items()
                            if k != "raw_generation_response"
                        }
                    )

            if self.labeller is not None:
                # `num_generations` is always 1 because labelling the same input multiple times
                # using the same LLM may not make sense
                batch_labels = self.labeller.generate(
                    inputs=inputs,
                    num_generations=1,
                    progress_callback_func=labelling_progress_bar,
                )
                labels.extend(batch_labels)

        if self.labeller is not None:
            # If the LLM returns futures, we need to wait for them to finish
            if self.labeller.return_futures:
                labels = [future.result() for future in labels]
            # TODO(alvarobartt): implement fallback mechanism if `combine_dicts` fails
            labels = [
                {
                    "raw_labelling_response": raw_response,
                    **combine_dicts(*parsed_response),
                }
                for raw_response, parsed_responses in labels
                for parsed_response in parsed_responses
            ]

        dataset = self._add_columns_to_dataset(dataset, generations, labels)
        dataset = self._remap_dataset(dataset)
        dataset.task = self.labeller.task
        return dataset


def pipeline(
    task: Literal["preference"],
    llm: Optional["LLM"] = None,
    openai_api_key: Optional[str] = None,
    **kwargs,
) -> "Pipeline":
    if task == "preference":
        from ultralabel.dataset import PreferenceDataset
        from ultralabel.llm.openai_ import OpenAILLM
        from ultralabel.tasks.openai_ import OpenAIResponseRating

        task_kwargs = {
            key: kwargs.get(key)
            for key in OpenAIResponseRating.__fields__.keys()
            if key in kwargs
        }
        labelling_llm = OpenAILLM(
            model=kwargs.get("openai_model") or "gpt-3.5-turbo",
            task=OpenAIResponseRating(**task_kwargs),
            max_new_tokens=kwargs.get("max_new_tokens") or 256,
            num_threads=kwargs.get("num_threads") or 4,
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=kwargs.get("temperature") or 0.0,
        )
        dataset_cls = PreferenceDataset
    else:
        raise ValueError(f"Invalid task: {task}")

    class CustomPipeline(Pipeline[dataset_cls]):
        pass

    CustomPipeline.dataset_cls = dataset_cls

    return CustomPipeline(generation_llm=llm, labelling_llm=labelling_llm)
