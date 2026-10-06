---
hide:
  - navigation
---
# UMAP

UMAP is a general purpose manifold learning and dimension reduction algorithm.



This is a `GlobalStep` that reduces the dimensionality of the embeddings using. Visit
    the `TextClustering` step for an example of use. The trained model is saved as an artifact
    when creating a distiset and pushing it to the Hugging Face Hub.





### Attributes

- **n_components**: The dimension of the space to embed into. This defaults to 2 to  provide easy visualization (that's probably what you want), but can  reasonably be set to any integer value in the range 2 to 100.  - metric: The metric to use to compute distances in high dimensional space.  Visit UMAP's documentation for more information. Defaults to `euclidean`.  - n_jobs: The number of parallel jobs to run. Defaults to `8`.  - random_state: The random state to use for the UMAP algorithm.




### Runtime Parameters

- **n_components**: The dimension of the space to embed into. This defaults to 2 to  provide easy visualization (that's probably what you want), but can  reasonably be set to any integer value in the range 2 to 100.

- **metric**: The metric to use to compute distances in high dimensional space.  Visit UMAP's documentation for more information. Defaults to `euclidean`.

- **n_jobs**: The number of parallel jobs to run. Defaults to `8`.

- **random_state**: The random state to use for the UMAP algorithm.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[embedding]
		end
		subgraph New columns
			OCOL0[projection]
		end
	end

	subgraph UMAP
		StepInput[Input Columns: embedding]
		StepOutput[Output Columns: projection]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **embedding** (`List[float]`): The original embeddings we want to reduce the dimension.




#### Outputs


- **projection** (`List[float]`): Embedding reduced to the number of components specified,  the size of the new embeddings will be determined by the `n_components`.







### References

- [UMAP repository](https://github.com/lmcinnes/umap/tree/master)

- [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/)


