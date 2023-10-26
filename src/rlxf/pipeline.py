from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from rlxf.dataset import CustomDataset
from rlxf.progress_bar import get_progress_bars_for_pipeline
from rlxf.utils import combine_dicts

if TYPE_CHECKING:
    from datasets import Dataset

    from rlxf.llm.base import LLM


class Pipeline:
    def __init__(
        self,
        generation_llm: Union["LLM", None] = None,
        labelling_llm: Union["LLM", None] = None,
    ) -> None:
        self.generation_llm = generation_llm
        self.labelling_llm = labelling_llm

        if self.generation_llm is None and self.labelling_llm is None:
            raise ValueError("At least one LLM has to be provided to the pipeline")

    def _remap_dataset(self, dataset: "Dataset") -> CustomDataset:
        # Dynamically remaps the `datasets.Dataset` to be a `CustomDataset`
        dataset.__class__ = CustomDataset
        return dataset  # type: ignore

    def _validate_dataset(self, dataset: "Dataset") -> None:
        # Generation LLM has not been provided, so the columns needed by the Labelling
        # LLM must be in the provided dataset
        if self.labelling_llm is not None:
            if self.generation_llm is None:
                try:
                    self.labelling_llm.prompt_template.validate_dataset(
                        dataset.column_names
                    )
                except KeyError as err:
                    raise KeyError(
                        f"Labelling LLM expects a dataset with the following columns: {self.labelling_llm.prompt_template.input_args_names}"
                    ) from err
            else:
                expected_columns = (
                    dataset.column_names
                    + self.generation_llm.prompt_template.output_args_names
                )
                try:
                    self.labelling_llm.prompt_template.validate_dataset(
                        expected_columns
                    )
                except KeyError as err:
                    raise KeyError(
                        f"Labelling LLM expects to receive the following columns after the generation process: {expected_columns}"
                    ) from err

        if self.generation_llm is not None:
            try:
                self.generation_llm.prompt_template.validate_dataset(
                    dataset.column_names
                )
            except KeyError as err:
                raise KeyError(
                    f"Generation LLM expects a dataset with the following columns: {self.generation_llm.prompt_template.input_args_names}"
                ) from err

    def _add_columns_to_dataset(
        self,
        dataset: "Dataset",
        generations: List[Dict[str, Any]],
        labels: List[Dict[str, Any]],
    ) -> "Dataset":
        if self.generation_llm is not None:
            for output_name in self.generation_llm.prompt_template.output_args_names:
                dataset = dataset.add_column(
                    output_name, [row.get(output_name, None) for row in generations]
                )

        if self.labelling_llm is not None:
            for output_name in self.labelling_llm.prompt_template.output_args_names:
                dataset = dataset.add_column(
                    output_name, [row.get(output_name, None) for row in labels]
                )

        return dataset

    def _get_batch_generations(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int,
        progress_callback_func: Union[Callable, None] = None,
    ) -> List[Dict[str, Any]]:
        batch_generations = self.generation_llm.generate(
            inputs=inputs,
            num_generations=num_generations,
            progress_callback_func=progress_callback_func,
        )

        if self.generation_llm.return_futures:
            batch_generations = [future.result() for future in batch_generations]

        return [
            combine_dicts(*input_generation) for input_generation in batch_generations
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
        dataset: "Dataset",
        num_generations: int = 1,
        batch_size: int = 1,
        display_progress_bar: bool = False,

    ) -> CustomDataset:
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

            if self.generation_llm is not None:
                batch_generations = self._get_batch_generations(
                    inputs, num_generations, generation_progress_func
                )
                generations.extend(batch_generations)

                for input, generations_ in zip(inputs, batch_generations):
                    input.update(generations_)

            if self.labelling_llm is not None:
                # `num_generations` is always 1 because labelling the same input multiple times
                # using the same LLM may not make sense
                batch_labels = self.labelling_llm.generate(
                    inputs=inputs, num_generations=1,
                    progress_callback_func=labelling_progress_bar
                )
                labels.extend(batch_labels)

        if self.labelling_llm is not None:
            # If the LLM returns futures, we need to wait for them to finish
            if self.labelling_llm.return_futures:
                labels = [future.result() for future in labels]
            labels = [
                combine_dicts(*label)
                for batch_labels in labels
                for label in batch_labels
            ]

        dataset = self._add_columns_to_dataset(dataset, generations, labels)
        dataset = self._remap_dataset(dataset)
        # TODO: remove once it's properly pre-defined
        # List of things we need to know in advance:
        # * Which are the fields for the `FeedbackDataset` and how can those be mapped with the LLM inputs?
        # * Which are the questions for the `FeedbackDataset` and how can those be mapped with the LLM outputs?
        # * At least one sample input and one sample output
        import argilla as rg

        # TODO: as there's just one field type i.e. `text`, these can be easily inferred from `input_args_names`
        fields = []
        # TODO: there's no way to infer those, unless those are pre-defined at `prompt_template` level [PROBLEMATIC]
        questions = []
        # TODO: `input_args` can be easily inferred from `input_args_names`
        input_args = {}
        # TODO: `output_args` can be easily inferred from the pre-defined questions
        output_args = {}
        for input_arg_name in self.labelling_llm.prompt_template.input_args_names:
            if isinstance(inputs[0][input_arg_name], list):
                for idx in range(1, len(inputs[0][input_arg_name]) + 1):
                    fields.append(rg.TextField(name=f"{input_arg_name}-{idx}"))
                    if input_arg_name not in input_args:
                        input_args[input_arg_name] = {}
                    input_args[input_arg_name].update(
                        {idx - 1: f"{input_arg_name}-{idx}"}
                    )
                    for (
                        output_arg_name
                    ) in self.labelling_llm.prompt_template.output_args_names:
                        if output_arg_name not in output_args:
                            output_args[output_arg_name] = {}
                        output_args[output_arg_name].update(
                            {idx - 1: f"{input_arg_name}-{idx}-{output_arg_name}"}
                        )
                    questions.extend(
                        [
                            rg.RatingQuestion(
                                name=f"{input_arg_name}-{idx}-rating",
                                title=f"Whats's the rating for {input_arg_name}-{idx}?",
                                values=list(
                                    range(
                                        1,
                                        len(self.labelling_llm.prompt_template.ranks)
                                        + 1,
                                    )
                                ),
                            ),
                            rg.TextQuestion(
                                name=f"{input_arg_name}-{idx}-rationale",
                                title=f"Whats's the rationale behind {input_arg_name}-{idx}'s rating?",
                            ),
                        ]
                    )
            else:
                # TODO: how do we define what's in and what's out of the questions?
                fields.append(rg.TextField(name=input_arg_name))
                input_args[input_arg_name] = input_arg_name

        dataset.argilla_fields = fields
        dataset.argilla_questions = questions
        dataset.argilla_input_args = input_args
        dataset.argilla_output_args = output_args
        return dataset
