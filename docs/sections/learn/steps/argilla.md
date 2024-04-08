# Argilla

As an additional step, besides being able to restore the dataset from the `Pipeline` output as a `Distiset` (which is a `datasets.DatasetDict` with multiple configurations depending on the leaf nodes of the `Pipeline`), one can also include a `Step` within the `Pipeline` to easily export the datasets to Argilla with a pre-defined configuration, suiting the annotation purposes.

Being able to export the generated synthetic datasets to Argilla, was one of the core features we wanted to have fully integrated within `distilabel`, not only because we're biased, but also because we believe in the potential of synthetic data, but always having an annotator or group of annotators to review the data and draw conclusions from it. So on, the Argilla integration will make it easier than ever to push a dataset to Argilla while the `Pipeline` is running, and be able to follow along the generation process, as well as annotating the records on the fly.

### Text Generation

For text generation scenarios, i.e. when the `Pipeline` contains a `TextGeneration` step, we have designed the task [`TextGenerationToArgilla`][distilabel.steps.argilla.text_generation], which will seamlessly push the generated data to Argilla, and allow the annotator to review the records.

The dataset will be pushed with the following configuration:

* Fields: `instruction` and `generation`, both being fields of type `argilla.TextField`, plus the automatically generated `id` for the given `instruction` to be able to search with it. The field `instruction` must always be a string, while the field `generation` can either be a single string or a list of strings (useful when there are multiple parent nodes of type `TextGeneration`); even though each record will always contain at most one `instruction`-`generation` pair.

* Questions: `quality` will be the only question for the annotators to answer i.e. to annotate, and it will be an `argilla.LabelQuestion` referring to the quality of the provided generation for the given instruction, and can be annotated with either 👎 (bad) or 👍 (good).

!!! NOTE
    The `TextGenerationToArgilla` step will only work as is if the `Pipeline` contains one or multiple `TextGeneration` steps, or if the columns `instruction` and `generation` are available within the batch data. Otherwise, the variable `input_mappings` will need to be set so that either both or one of `instruction` and `generation` are mapped to one of the existing columns in the batch data.

```python
from distilabel.llms import OpenAILLM
from distilabel.steps import LoadDataFromDicts, PreferenceToArgilla
from distilabel.steps.tasks import TextGeneration


with Pipeline(name="my-pipeline") as pipeline:
    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=[
            {
                "instruction": "Write a short story about a dragon that saves a princess from a tower.",
            },
        ],
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=OpenAILLM(model="gpt-4"),
    )
    load_dataset.connect(text_generation)

    to_argilla = TextGenerationToArgilla(
        dataset_name="my-dataset",
        dataset_workspace="admin",
        api_url="<ARGILLA_API_URL>",
        api_key="<ARGILLA_API_KEY>",
    )

    text_generation.connect(to_argilla)

pipeline.run()
```

### Preference

For preference scenarios, i.e. when the `Pipeline` contains multiple `TextGeneration` steps, we have designed the task [`PreferenceToArgilla`][distilabel.steps.argilla.preference], which will seamlessly push the generated data to Argilla, and allow the annotator to review the records.

The dataset will be pushed with the following configuration:

* Fields: `instruction` and `generations`, both being fields of type `argilla.TextField`, plus the automatically generated `id` for the given `instruction` to be able to search with it. The field `instruction` must always be a string, while the field `generations` must be a list of strings, containing the generated texts for the given `instruction` so that at least there are two generations to compare. Other than that, the number of `generation` fields within each record in Argilla will be defined by the value of the variable `num_generations` to be provided in the `PreferenceToArgilla` step.

* Questions: `rating` and `rationale` will be the pairs of questions to be defined per each generation i.e. per each value within the range from 0 to `num_generations`, and those will be of types `argilla.RatingQuestion` and `argilla.TextQuestion`, respectively. Also note that only the first pair of questions will be mandatory, since only one generation is ensured to be within the batch data. Additionally, note that the provided ratings will range from 1 to 5, and to mention that Argilla only supports values above 0.

!!! NOTE
    The `PreferenceToArgilla` step will only work as is if the `Pipeline` contains multiple `TextGeneration` steps, or if the columns `instruction` and `generations` are available within the batch data. Otherwise, the variable `input_mappings` will need to be set so that either both or one of `instruction` and `generations` are mapped to one of the existing columns in the batch data.

!!! NOTE
    Additionally, if the `Pipeline` contains an `UltraFeedback` step, the `ratings` and `rationales` will also be available, so if that's the case, those will be automatically injected as suggestions to the existing dataset so that the annotator only needs to review those, instead of fulfilling those by themselves.

```python
from distilabel.llms import OpenAILLM
from distilabel.steps import LoadDataFromDicts, PreferenceToArgilla
from distilabel.steps.tasks import TextGeneration


with Pipeline(name="my-pipeline") as pipeline:
    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=[
            {
                "instruction": "Write a short story about a dragon that saves a princess from a tower.",
            },
        ],
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=OpenAILLM(model="gpt-4"),
        num_generations=4,
        group_generations=True,
    )
    load_dataset.connect(text_generation)

    to_argilla = PreferenceToArgilla(
        dataset_name="my-dataset",
        dataset_workspace="admin",
        api_url="<ARGILLA_API_URL>",
        api_key="<ARGILLA_API_KEY>",
        num_generations=4,
    )
    text_generation.connect(to_argilla)

pipeline.run()
```

!!! NOTE
    If you are willing to also add the suggestions, feel free to check [UltraFeedback: Boosting Language Models with High-quality Feedback](../../papers/ultrafeedback.md) where the `UltraFeedback` task is used to generate both ratings and rationales for each of the generations of a given instruction.
