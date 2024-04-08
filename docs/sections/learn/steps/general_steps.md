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
