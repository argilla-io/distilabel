# Generator Steps

This section shows a special type of step that don't need a prior step to generate data, the generator steps, which will be in charge of loading and yielding data.

## Just load some data

The easiest way to create a step that generates some data is to pass some dict with the fields and the data we want. This is what [`LoadDataFromDicts`][distilabel.steps.generators.data] does for us. Let's see an example of how to instantiate it with a couple of examples of instruction/completion pairs:

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
    batch_size=1,
    pipeline=Pipeline(name="data-pipeline")
)
```

As we can see, this step doesn't have much processing logic to do, it will generate data with the batch size we tell it to:

```python
>>> next(load_data.process())
([{'instruction': 'What if the Beatles had never formed as a band?', 'completion': 'The Beatles are widely credited with starting the British Invasion, a wave of rock and roll bands from the United Kingdom that became popular in America in the 1960s. If the Beatles had never formed, this musical movement may have never happened, and the world may have been exposed to a different kind of music. So, without the Beatles’ fame and success, other bands wouldn’t have been able to break into the American music scene and create a wider range of sounds. We could have ended up with a less interesting mix of songs playing on the radio.'}], False)
```

It will yield `GeneratorStepOutput` objects, an iterator of tuples where the first element is the batch of data, and the second is a boolean flag indicating whether this batch is the last one (to internally determine when to stop yielding data).

## Loading structured data

Unless we are doing some testing, we are more likely going to work with a proper dataset:

### Load a dataset from Hugging Face Hub

The easiest way to ingest data from a dataset is using the [`LoadHubDataset`][distilabel.steps.generators.huggingface] step, let's see an example:

```python
from distilabel.steps.generators.huggingface import LoadHubDataset

load_hub_dataset = LoadHubDataset(
    name="load_dataset",
    repo_id="distilabel-internal-testing/instruction-dataset-mini",
    split="test",
    batch_size=2,
    pipeline=Pipeline(name="dataset-pipeline")    
)
load_hub_dataset.load()
```

We see that creating a step to load a dataset from the Hugging Face Hub is almost the same as loading it directly using `datasets.load_dataset`, with one remark, we have to call `.load()` on our step. The reason for this extra step is because internally we want to do the actual processing at the correct moment in the whole pipeline, we don't just need to take care of this call because we are working with it outside of a `Pipeline`.

And let's request the following batch:

```python
>>> next(load_hub_dataset.process())
([{'prompt': 'Arianna has 12 chocolates more than Danny. Danny has 6 chocolates more than Robbie. Arianna has twice as many chocolates as Robbie has. How many chocolates does Danny have?', 'completion': 'Denote the number of chocolates each person has by the letter of their first name. We know that\nA = D + 12\nD = R + 6\nA = 2 * R\n\nThus, A = (R + 6) + 12 = R + 18\nSince also A = 2 * R, this means 2 * R = R + 18\nHence R = 18\nHence D = 18 + 6 = 24', 'meta': {'id': 0, 'motivation_app': None, 'prompt': 'Arianna has 12 chocolates more than Danny. Danny has 6 chocolates more than Robbie. Arianna has twice as many chocolates as Robbie has. How many chocolates does Danny have?', 'input': None, 'completion': 'Denote the number of chocolates each person has by the letter of their first name. We know that\nA = D + 12\nD = R + 6\nA = 2 * R\n\nThus, A = (R + 6) + 12 = R + 18\nSince also A = 2 * R, this means 2 * R = R + 18\nHence R = 18\nHence D = 18 + 6 = 24', 'source': 'surge', 'category': 'Question Answering', 'subcategory': 'Math'}}, {'prompt': 'Write a plot summary for a comedic novel involving Elon Musk and sea travel.', 'completion': 'Elon Musk hires a team of experts to build the ultimate yacht, but when the yacht is completed, he realizes that he has no idea how to sail it. With the help of a quirky crew and a fearless captain, the playboy embarks on a wild and hilarious adventure across the open seas, where the crew have to keep Elon alive despite his inability to do anything himself. All the while, Elon takes credit for their hard work.', 'meta': {'id': 1, 'motivation_app': None, 'prompt': 'Write a plot summary for a comedic novel involving Elon Musk and sea travel.', 'input': None, 'completion': 'Elon Musk hires a team of experts to build the ultimate yacht, but when the yacht is completed, he realizes that he has no idea how to sail it. With the help of a quirky crew and a fearless captain, the playboy embarks on a wild and hilarious adventure across the open seas, where the crew have to keep Elon alive despite his inability to do anything himself. All the while, Elon takes credit for their hard work.', 'source': 'surge', 'category': 'Generation', 'subcategory': 'Story generation'}}], False)
```

We can see the same structure (for a different type of dataset) as we had with the simpler `LoadDataFromDicts`.
