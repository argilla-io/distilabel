# GenerateLongTextMatchingData


Generate long text matching data with an `LLM` to later on train an embedding model.



`GenerateLongTextMatchingData` is a `Task` that generates long text matching data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.



### Note
Ideally this task should be used with `EmbeddingTaskGenerator` with `flatten_tasks=True`
with the `category="text-matching-long"`; so that the `LLM` generates a list of tasks that
are flattened so that each row contains a single task for the text-matching-long category.



### Attributes

- **language**: The language of the data to be generated, which can be any of the languages  retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.

- **seed**: The random seed to be set in case there's any sampling within the `format_input` method.  Note that in this task the `seed` has no effect since there are no sampling params.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
	end

	subgraph GenerateLongTextMatchingData
	end


```







### Examples


#### Generate synthetic long text matching data for training embedding models
```python
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EmbeddingTaskGenerator, GenerateLongTextMatchingData

with Pipeline("my-pipeline") as pipeline:
    task = EmbeddingTaskGenerator(
        category="text-matching-long",
        flatten_tasks=True,
        llm=...,  # LLM instance
    )

    generate = GenerateLongTextMatchingData(
        language="English",
        llm=...,  # LLM instance
    )

    task >> generate
```




### References

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)


