from __future__ import annotations

from typing import TYPE_CHECKING

from rlxf.utils import combine_dicts

if TYPE_CHECKING:
    from datasets import Dataset

    from rlxf.llm.base import LLM


class Pipeline:
    def __init__(self, generation_llm: LLM, labelling_llm: LLM) -> None:
        self.generation_llm = generation_llm
        self.labelling_llm = labelling_llm

    def _validate_dataset(self, dataset: Dataset) -> None:
        for input_arg_name in self.generation_llm.prompt_template.input_args_names:
            if input_arg_name not in dataset.column_names:
                raise ValueError(
                    f"Generation LLM expects a column named '{input_arg_name}' in the"
                    "provided dataset, but it was not found."
                )

    def _add_columns_to_dataset(self, dataset: Dataset, generations, labels) -> Dataset:
        for output_name in self.generation_llm.prompt_template.output_args_names:
            dataset = dataset.add_column(
                output_name, [row[output_name] for row in generations]
            )

        for output_name in self.labelling_llm.prompt_template.output_args_names:
            dataset = dataset.add_column(
                output_name, [row[output_name] for row in labels]
            )

        return dataset

    def generate(
        self, dataset: Dataset, num_generations: int = 1, batch_size: int = 1
    ) -> Dataset:
        self._validate_dataset(dataset)

        generations = []
        labels = []

        for rows in dataset.iter(batch_size=batch_size):
            inputs = []
            for col, values in rows.items():
                inputs.extend([{col: value} for value in values])

            batch_generations = self.generation_llm.generate(
                inputs, num_generations=num_generations
            )

            if self.generation_llm.return_futures:
                batch_generations = [future.result() for future in batch_generations]

            batch_generations = [
                combine_dicts(*input_generation)
                for input_generation in batch_generations
            ]

            generations.extend(batch_generations)

            for input, generations_ in zip(inputs, batch_generations):
                input.update(generations_)

            batch_labels = self.labelling_llm.generate(inputs)
            labels.extend(batch_labels)

        # If the LLM returns futures, we need to wait for them to finish
        if self.labelling_llm.return_futures:
            labels = [future.result() for future in labels]
        labels = [
            combine_dicts(*label) for batch_labels in labels for label in batch_labels
        ]

        return self._add_columns_to_dataset(dataset, generations, labels)
