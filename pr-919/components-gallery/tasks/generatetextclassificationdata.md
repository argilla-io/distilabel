---
hide:
  - navigation
---
# GenerateTextClassificationData

Generate text classification data with an `LLM` to later on train an embedding model.



`GenerateTextClassificationData` is a `Task` that generates text classification data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.



### Note
Ideally this task should be used with `EmbeddingTaskGenerator` with `flatten_tasks=True`
with the `category="text-classification"`; so that the `LLM` generates a list of tasks that
are flattened so that each row contains a single task for the text-classification category.



### Attributes

- **language**: The language of the data to be generated, which can be any of the languages  retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.

- **difficulty**: The difficulty of the query to be generated, which can be `high school`, `college`, or `PhD`.  Defaults to `None`, meaning that it will be randomly sampled.

- **clarity**: The clarity of the query to be generated, which can be `clear`, `understandable with some effort`,  or `ambiguous`. Defaults to `None`, meaning that it will be randomly sampled.

- **seed**: The random seed to be set in case there's any sampling within the `format_input` method.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
	end

	subgraph GenerateTextClassificationData
	end


```







### Examples


#### Generate synthetic text classification data for training embedding models
```python
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EmbeddingTaskGenerator, GenerateTextClassificationData

with Pipeline("my-pipeline") as pipeline:
    task = EmbeddingTaskGenerator(
        category="text-classification",
        flatten_tasks=True,
        llm=...,  # LLM instance
    )

    generate = GenerateTextClassificationData(
        language="English",
        difficulty="high school",
        clarity="clear",
        llm=...,  # LLM instance
    )

    task >> generate
```




### References

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)


