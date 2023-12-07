# Pipelines

The `Pipeline` class is the center piece of `distilabel`, in charge of the generation and labelling process. 

We will use a sample of a instruction[^1] dataset to guide us through the process. 

[^1]:
    You can take a look at [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) to get acquainted with instruction-following datasets.

Let's start by loading it and importing the necessary code

```python
from datasets import load_dataset

dataset = load_dataset("argilla/distilabel-docs", split="train")
dataset = dataset.remove_columns(
    [column for column in dataset.column_names if column not in ["input"]]
)
# >>> dataset[0]["input"]
# 'Arianna has 12 chocolates more than Danny. Danny has 6 chocolates more than Robbie. Arianna has twice as many chocolates as Robbie has. How many chocolates does Danny have?'
```



```python
from distilabel.llm import LlamaCppLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask
from llama_cpp import Llama

```

The API reference can be found here: [Pipeline][distilabel.pipeline.Pipeline]

The API reference can be found here: [pipeline][distilabel.pipeline.pipeline]
