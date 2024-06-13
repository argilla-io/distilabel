---
hide:
  - navigation
---
# MonolingualTripletGenerator

Generate monolingual triplets with an `LLM` to later on train an embedding model.



`MonolingualTripletGenerator` is a `GeneratorTask` that generates monolingual triplets with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.





### Attributes

- **language**: The language of the data to be generated, which can be any of the languages  retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.

- **unit**: The unit of the data to be generated, which can be `sentence`, `phrase`, or `passage`.  Defaults to `None`, meaning that it will be randomly sampled.

- **difficulty**: The difficulty of the query to be generated, which can be `elementary school`, `high school`, or `college`.  Defaults to `None`, meaning that it will be randomly sampled.

- **high_score**: The high score of the query to be generated, which can be `4`, `4.5`, or `5`.  Defaults to `None`, meaning that it will be randomly sampled.

- **low_score**: The low score of the query to be generated, which can be `2.5`, `3`, or `3.5`.  Defaults to `None`, meaning that it will be randomly sampled.

- **seed**: The random seed to be set in case there's any sampling within the `format_input` method.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
	end

	subgraph MonolingualTripletGenerator
	end


```







### Examples


#### Generate monolingual triplets for training embedding models
```python
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MonolingualTripletGenerator

with Pipeline("my-pipeline") as pipeline:
    task = MonolingualTripletGenerator(
        language="English",
        unit="sentence",
        difficulty="elementary school",
        high_score="4",
        low_score="2.5",
        llm=...,
    )

    ...

    task >> ...
```




