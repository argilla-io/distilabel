---
hide:
  - navigation
---
# FaissNearestNeighbour

Create a `faiss` index to get the nearest neighbours.



`FaissNearestNeighbour` is a `GlobalStep` that creates a `faiss` index using the Hugging
    Face `datasets` library integration, and then gets the nearest neighbours and the scores
    or distance of the nearest neighbours for each input row.





### Attributes

- **device**: the CUDA device ID or a list of IDs to be used. If negative integer, it  will use all the available GPUs. Defaults to `None`.

- **string_factory**: the name of the factory to be used to build the `faiss` index.  Available string factories can be checked here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes.  Defaults to `None`.

- **metric_type**: the metric to be used to measure the distance between the points. It's  an integer and the recommend way to pass it is importing `faiss` and then passing  one of `faiss.METRIC_x` variables. Defaults to `None`.

- **k**: the number of nearest neighbours to search for each input row. Defaults to `1`.

- **search_batch_size**: the number of rows to include in a search batch. The value can  be adjusted to maximize the resources usage or to avoid OOM issues. Defaults  to `50`.




### Runtime Parameters

- **device**: the CUDA device ID or a list of IDs to be used. If negative integer,  it will use all the available GPUs. Defaults to `None`.

- **string_factory**: the name of the factory to be used to build the `faiss` index.  Available string factories can be checked here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes.  Defaults to `None`.

- **metric_type**: the metric to be used to measure the distance between the points.  It's an integer and the recommend way to pass it is importing `faiss` and then  passing one of `faiss.METRIC_x` variables. Defaults to `None`.

- **k**: the number of nearest neighbours to search for each input row. Defaults to `1`.

- **search_batch_size**: the number of rows to include in a search batch. The value  can be adjusted to maximize the resources usage or to avoid OOM issues. Defaults  to `50`.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[embedding]
		end
		subgraph New columns
			OCOL0[nn_indices]
			OCOL1[nn_scores]
		end
	end

	subgraph FaissNearestNeighbour
		StepInput[Input Columns: embedding]
		StepOutput[Output Columns: nn_indices, nn_scores]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **embedding** (`List[Union[float, int]]`): a sentence embedding.




#### Outputs


- **nn_indices** (`List[int]`): a list containing the indices of the `k` nearest neighbours  in the inputs for the row.

- **nn_scores** (`List[float]`): a list containing the score or distance to each `k`  nearest neighbour in the inputs.





### Examples


#### Generating embeddings and getting the nearest neighbours
```python
from distilabel.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from distilabel.pipeline import Pipeline
from distilabel.steps import EmbeddingGeneration, FaissNearestNeighbour, LoadDataFromHub

with Pipeline(name="hello") as pipeline:
    load_data = LoadDataFromHub(output_mappings={"prompt": "text"})

    embeddings = EmbeddingGeneration(
        embeddings=SentenceTransformerEmbeddings(
            model="mixedbread-ai/mxbai-embed-large-v1"
        )
    )

    nearest_neighbours = FaissNearestNeighbour()

    load_data >> embeddings >> nearest_neighbours

if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            load_data.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
        },
        use_cache=False,
    )
```




### References

- [The Faiss library](https://arxiv.org/abs/2401.08281)


