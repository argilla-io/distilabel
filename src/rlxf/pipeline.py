from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

    from rlxf.llm.base import LLM


class Pipeline:
    def __init__(self, generation_llm: LLM, labelling_llm: LLM) -> None:
        self.generation_llm = generation_llm
        self.labelling_llm = labelling_llm

    def generate(
        self, dataset: Dataset, num_generations: int = 1, batch_size: int = 1
    ) -> Dataset:
        generations = []
        labels = []

        for rows in dataset.iter(batch_size=batch_size):
            inputs = []
            for col, values in rows.items():
                inputs.extend([{col: value} for value in values])

            batch_generations = self.generation_llm.generate(
                inputs, num_generations=num_generations
            )
            generations.extend(batch_generations)

            for input, generations in zip(inputs, batch_generations):
                input["responses"] = generations

            batch_labels = self.labelling_llm.generate(inputs)
            labels.extend(batch_labels)

        dataset = dataset.add_column("generation", generations).add_column("labels", labels)

        return dataset
