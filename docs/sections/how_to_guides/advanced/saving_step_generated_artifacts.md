# Saving step generated artifacts

Some `Step`s might need to produce an auxiliary artifact that is not a result of the computation, but is needed for the computation. For example, the [`FaissNearestNeighbour`](../../../components-gallery/steps/faissnearestneighbour.md) needs to create a Faiss index to compute the output of the step which are the top `k` nearest neighbours for each input. Generating the Faiss index takes time and it could potentially be reused outside of the `distilabel` pipeline, so it would be a shame not saving it.

For this reason, `Step`s have a method called `save_artifact` that allows saving artifacts that will be included along the outputs of the pipeline in the generated [`Distiset`][distilabel.distiset.Distiset]. The generated artifacts will be uploaded and saved when using `Distiset.push_to_hub` or `Distiset.save_to_disk` respectively. Let's see how to use it with a simple example.

```python
from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt

from distilabel.steps import GlobalStep, StepInput, StepOutput
from distilabel.steps.typing import StepColumns

if TYPE_CHECKING:
    from distilabel.steps import StepOutput


class CountTextCharacters(GlobalStep):
    inputs: StepColumns = ["text"]
    outputs: StepColumns = ["text_character_count"]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        character_counts = []

        for input in inputs:
            text_character_count = len(input["text"])
            input["text_character_count"] = text_character_count
            character_counts.append(text_character_count)

        # Generate plot with the distribution of text character counts
        plt.figure(figsize=(10, 6))
        plt.hist(character_counts, bins=30, edgecolor="black")
        plt.title("Distribution of Text Character Counts")
        plt.xlabel("Character Count")
        plt.ylabel("Frequency")

        # Save the plot as an artifact of the step
        self.save_artifact(
            name="text_character_count_distribution",
            write_function=lambda path: plt.savefig(path / "figure.png"),
            metadata={"type": "image", "library": "matplotlib"},
        )

        plt.close()

        yield inputs
```

As it can be seen in the example above, we have created a simple step that counts the number of characters in each input text and generates a histogram with the distribution of the character counts. We save the histogram as an artifact of the step using the `save_artifact` method. The method takes three arguments:

- `name`: The name we want to give to the artifact.
- `write_function`: A function that writes the artifact to the desired path. The function will receive a `path` argument which is a `pathlib.Path` object pointing to the directory where the artifact should be saved.
- `metadata`: A dictionary with metadata about the artifact. This metadata will be saved along with the artifact.

Let's execute the step with a simple pipeline and push the resulting `Distiset` to the Hugging Face Hub:

??? "Example full code"

    ```python
    from typing import TYPE_CHECKING, List

    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from distilabel.pipeline import Pipeline
    from distilabel.steps import GlobalStep, StepInput, StepOutput
    from distilabel.steps.typing import StepColumns

    if TYPE_CHECKING:
        from distilabel.steps import StepOutput


    class CountTextCharacters(GlobalStep):
        inputs: StepColumns = ["text"]
        outputs: StepColumns = ["text_character_count"]

        def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
            character_counts = []

            for input in inputs:
                text_character_count = len(input["text"])
                input["text_character_count"] = text_character_count
                character_counts.append(text_character_count)

            # Generate plot with the distribution of text character counts
            plt.figure(figsize=(10, 6))
            plt.hist(character_counts, bins=30, edgecolor="black")
            plt.title("Distribution of Text Character Counts")
            plt.xlabel("Character Count")
            plt.ylabel("Frequency")

            # Save the plot as an artifact of the step
            self.save_artifact(
                name="text_character_count_distribution",
                write_function=lambda path: plt.savefig(path / "figure.png"),
                metadata={"type": "image", "library": "matplotlib"},
            )

            plt.close()

            yield inputs


    with Pipeline() as pipeline:
        count_text_characters = CountTextCharacters()

    if __name__ == "__main__":
        distiset = pipeline.run(
            dataset=load_dataset(
                "HuggingFaceH4/instruction-dataset", split="test"
            ).rename_column("prompt", "text"),
        )

        distiset.push_to_hub("distilabel-internal-testing/distilabel-artifacts-example")
    ```

The generated [distilabel-internal-testing/distilabel-artifacts-example](https://huggingface.co/datasets/distilabel-internal-testing/distilabel-artifacts-example) dataset repository has a section in its card [describing the artifacts generated by the pipeline](https://huggingface.co/datasets/distilabel-internal-testing/distilabel-artifacts-example#artifacts) and the generated plot can be seen [here](https://huggingface.co/datasets/distilabel-internal-testing/distilabel-artifacts-example/blob/main/artifacts/count_text_characters_0/text_character_count_distribution/figure.png).
