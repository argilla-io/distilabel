# General Steps

This section shows some steps that don't belong to the special category of `global` or `generator` steps.

## Processing columns of the dataset

The following steps correspond to some common operations that can be helpful during the dataset generation.

!!! Note
    We will use a sample dataset from a dict, keep in mind that as we are working with iterators (note the call to `next` function), you may need to recreate the dataset
    to execute all the steps at once.

```python
from distilabel.pipeline.local import Pipeline
from distilabel.steps.generators.data import LoadDataFromDicts

load_data = LoadDataFromDicts(
    name="load_data",
    data=[
        {
            "instruction": "What if the Beatles had never formed as a band?",
            "completion": "The Beatles are widely credited with starting the British Invasion, a wave of rock and roll bands from the United Kingdom that became popular in America in the 1960s. If the Beatles had never formed, this musical movement may have never happened, and the world may have been exposed to a different kind of music. So, without the Beatles\u2019 fame and success, other bands wouldn\u2019t have been able to break into the American music scene and create a wider range of sounds. We could have ended up with a less interesting mix of songs playing on the radio."
        },
        {
            "instruction": "Given that f(x) = 5x^3 - 2x + 3, find the value of f(2).",
            "completion": "The problem is asking us to find the value of the function f(x) = 5x^3 - 2x + 3 at the point x = 2. \n\nStep 1: Substitute x with 2 in the function\nWe replace every x in the function with 2. This gives us:\nf(2) = 5(2)^3 - 2(2) + 3\n\nStep 2: Simplify the expression\nNext, we simplify the expression by performing the operations in order from left to right.\n\nFirst, calculate the cube of 2, which is 8. Substitute this back into the expression:\nf(2) = 5(8) - 4 + 3\n\nThen, multiply 5 by 8 which gives us 40:\nf(2) = 40 - 4 + 3\n\nFinally, subtract 4 from 40 which gives us 36, and then add 3 to that:\nf(2) = 36 + 3\n\nStep 3: Final calculation\nNow, add 36 and 3 together:\nf(2) = 39\n\nSo, the value of the function f(x) = 5x^3 - 2x + 3 at the point x = 2 is 39."
        }
    ],
    pipeline=Pipeline(name="data-pipeline")
)
```

### Keep Columns

There is a special step to keep only the specified columns after a processing step: [`KeepColumns`][distilabel.steps.keep]. Let's use it to keep only the `instruction` column from the previous dataset:

```python
from distilabel.pipeline.local import Pipeline
from distilabel.steps.keep import KeepColumns

keep_columns = KeepColumns(
    name="keep-columns",
    columns=["instruction"],
    pipeline=Pipeline(name="keeper-pipeline"),
)
```

And to see it in action, let's grab the first batch of data:

```python
batch = next(load_data.process())[0]
print(json.dumps(next(keep_columns.process(batch)), indent=2))
# [
#   {
#     "instruction": "What if the Beatles had never formed as a band?"
#   },
#   {
#     "instruction": "Given that f(x) = 5x^3 - 2x + 3, find the value of f(2)."
#   }
# ]
```

After this step has processed the batch we have lost the `completion` column. This step can be useful to just keep the relevant columns after a step that generates some intermediate steps for example.

### Combine Columns

This next step allows us to merge the output from multiple steps into a single row for further processing, let's take a look at [`CombineColumns`][distilabel.steps.combine]:

```python
from distilabel.pipeline.local import Pipeline
from distilabel.steps.combine import CombineColumns

combine_columns = CombineColumns(
    name="combine_columns",
    columns=["instruction", "completion"],
    pipeline=Pipeline(name="combine-pipeline"),
)
```

To see the step in action, we are going to pass the previous batch as individual lists per row, mimicking what we would see during a pipeline in which we are combining the output from two different steps that could be generating data. We can understand each of these `[batch[i]]` as if it was the result from two different steps generating data:

```python
batch = next(load_data.process())[0]
combined = next(combine_columns.process([batch[0]], [batch[1]]))
print(json.dumps(combined, indent=2))
# [
#   {
#     "merged_instruction": [
#       "What if the Beatles had never formed as a band?",
#       "Given that f(x) = 5x^3 - 2x + 3, find the value of f(2)."
#     ],
#     "merged_completion": [
#       "The Beatles are widely credited with starting the British Invasion, a wave of rock and roll bands from the United Kingdom that became popular in America in the 1960s. If the Beatles had never formed, this musical movement may have never happened, and the world may have been exposed to a different kind of music. So, without the Beatles\u2019 fame and success, other bands wouldn\u2019t have been able to break into the American music scene and create a wider range of sounds. We could have ended up with a less interesting mix of songs playing on the radio.",
#       "The problem is asking us to find the value of the function f(x) = 5x^3 - 2x + 3 at the point x = 2. \n\nStep 1: Substitute x with 2 in the function\nWe replace every x in the function with 2. This gives us:\nf(2) = 5(2)^3 - 2(2) + 3\n\nStep 2: Simplify the expression\nNext, we simplify the expression by performing the operations in order from left to right.\n\nFirst, calculate the cube of 2, which is 8. Substitute this back into the expression:\nf(2) = 5(8) - 4 + 3\n\nThen, multiply 5 by 8 which gives us 40:\nf(2) = 40 - 4 + 3\n\nFinally, subtract 4 from 40 which gives us 36, and then add 3 to that:\nf(2) = 36 + 3\n\nStep 3: Final calculation\nNow, add 36 and 3 together:\nf(2) = 39\n\nSo, the value of the function f(x) = 5x^3 - 2x + 3 at the point x = 2 is 39."
#     ]
#   }
# ]
```

We have both `instruction` and `completion` from the 2 different lists merged as a single column: `merged_instruction` and `merged_completion` respectively.

This step is necessary to build more complicated pipelines like `UltraFeedback`, where we need to have the merged content of multiple `LLMs` to rate them.

### Expand Columns

Just as we may have the necessity to merge the output from different steps, we can equally want to *expand* the current columns to behave as multiple rows, let's see the [`ExpandColumns`][distilabel.steps.expand] work on the output from the previous step:

```python
from distilabel.pipeline.local import Pipeline
from distilabel.steps.expand import ExpandColumns

expand_columns = ExpandColumns(
    name="expand_columns",
    columns=["merged_instruction", "merged_completion"],
    pipeline=Pipeline(name="expand-pipeline"),
)
```

We can pass to the process method the `combined` variable, which is the output from the previous step directly:

```python
print(json.dumps(next(expand_columns.process(combined)), indent=2))
# [
#   {
#     "merged_instruction": "What if the Beatles had never formed as a band?",
#     "merged_completion": "The Beatles are widely credited with starting the British Invasion, a wave of rock and roll bands from the United Kingdom that became popular in America in the 1960s. If the Beatles had never formed, this musical movement may have never happened, and the world may have been exposed to a different kind of music. So, without the Beatles\u2019 fame and success, other bands wouldn\u2019t have been able to break into the American music scene and create a wider range of sounds. We could have ended up with a less interesting mix of songs playing on the radio."
#   },
#   {
#     "merged_instruction": "Given that f(x) = 5x^3 - 2x + 3, find the value of f(2).",
#     "merged_completion": "The problem is asking us to find the value of the function f(x) = 5x^3 - 2x + 3 at the point x = 2. \n\nStep 1: Substitute x with 2 in the function\nWe replace every x in the function with 2. This gives us:\nf(2) = 5(2)^3 - 2(2) + 3\n\nStep 2: Simplify the expression\nNext, we simplify the expression by performing the operations in order from left to right.\n\nFirst, calculate the cube of 2, which is 8. Substitute this back into the expression:\nf(2) = 5(8) - 4 + 3\n\nThen, multiply 5 by 8 which gives us 40:\nf(2) = 40 - 4 + 3\n\nFinally, subtract 4 from 40 which gives us 36, and then add 3 to that:\nf(2) = 36 + 3\n\nStep 3: Final calculation\nNow, add 36 and 3 together:\nf(2) = 39\n\nSo, the value of the function f(x) = 5x^3 - 2x + 3 at the point x = 2 is 39."
#   }
# ]
```

Obtaining the columns as a list of rows, that could be processed for a further step requiring the data in that special format.

## Uploading data to Argilla

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
from distilabel.steps import TextGenerationToArgilla


pipeline = Pipeline(name="my-pipeline")

...

step = TextGenerationToArgilla(
    dataset_name="my-dataset",
    dataset_workspace="admin",
    api_url="<ARGILLA_API_URL>",
    api_key="<ARGILLA_API_KEY>",
    pipeline=pipeline,
)

...
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
from distilabel.steps import TextGenerationToArgilla


pipeline = Pipeline(name="my-pipeline")

...

step = TextGenerationToArgilla(
    dataset_name="my-dataset",
    dataset_workspace="admin",
    api_url="<ARGILLA_API_URL>",
    api_key="<ARGILLA_API_KEY>",
    pipeline=pipeline,
)
...
```
