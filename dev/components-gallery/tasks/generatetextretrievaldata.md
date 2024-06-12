# GenerateTextRetrievalData


Generate text retrieval data with an `LLM` to later on train an embedding model.



`GenerateTextRetrievalData` is a `Task` that generates text retrieval data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.



### Note
Ideally this task should be used with `EmbeddingTaskGenerator` with `flatten_tasks=True`
with the `category="text-retrieval"`; so that the `LLM` generates a list of tasks that
are flattened so that each row contains a single task for the text-retrieval category.



### Attributes

- **language**: The language of the data to be generated, which can be any of the languages  retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.

- **query_type**: The type of query to be generated, which can be `extremely long-tail`, `long-tail`,  or `common`. Defaults to `None`, meaning that it will be randomly sampled.

- **query_length**: The length of the query to be generated, which can be `less than 5 words`, `5 to 15 words`,  or `at least 10 words`. Defaults to `None`, meaning that it will be randomly sampled.

- **difficulty**: The difficulty of the query to be generated, which can be `high school`, `college`, or `PhD`.  Defaults to `None`, meaning that it will be randomly sampled.

- **clarity**: The clarity of the query to be generated, which can be `clear`, `understandable with some effort`,  or `ambiguous`. Defaults to `None`, meaning that it will be randomly sampled.

- **num_words**: The number of words in the query to be generated, which can be `50`, `100`, `200`, `300`, `400`, or `500`.  Defaults to `None`, meaning that it will be randomly sampled.

- **seed**: The random seed to be set in case there's any sampling within the `format_input` method.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
	end

	subgraph GenerateTextRetrievalData
	end


```







### Examples


#### Generate synthetic text retrieval data for training embedding models
```python
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EmbeddingTaskGenerator, GenerateTextRetrievalData

with Pipeline("my-pipeline") as pipeline:
    task = EmbeddingTaskGenerator(
        category="text-retrieval",
        flatten_tasks=True,
        llm=...,  # LLM instance
    )

    generate = GenerateTextRetrievalData(
        language="English",
        query_type="common",
        query_length="5 to 15 words",
        difficulty="high school",
        clarity="clear",
        num_words=100,
        llm=...,  # LLM instance
    )

    task >> generate
```




### References

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)


