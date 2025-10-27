---
hide:
  - navigation
---
# DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core



samples in regions of high density and expands clusters from them. This algorithm
    is good for data which contains clusters of similar density.

    This is a `GlobalStep` that clusters the embeddings using the DBSCAN algorithm
    from `sklearn`. Visit `TextClustering` step for an example of use.
    The trained model is saved as an artifact when creating a distiset
    and pushing it to the Hugging Face Hub.





### Attributes

- **eps**: The maximum distance between two samples for one to be considered as in the  neighborhood of the other. This is not a maximum bound on the distances of  points within a cluster. This is the most important DBSCAN parameter to  choose appropriately for your data set and distance function.  - min_samples: The number of samples (or total weight) in a neighborhood for a point  to be considered as a core point. This includes the point itself. If `min_samples`  is set to a higher value, DBSCAN will find denser clusters, whereas if it is set  to a lower value, the found clusters will be more sparse.  - metric: The metric to use when calculating distance between instances in a feature  array. If metric is a string or callable, it must be one of the options allowed  by `sklearn.metrics.pairwise_distances` for its metric parameter.  - n_jobs: The number of parallel jobs to run.




### Runtime Parameters

- **eps**: The maximum distance between two samples for one to be considered as in the  neighborhood of the other. This is not a maximum bound on the distances of  points within a cluster. This is the most important DBSCAN parameter to  choose appropriately for your data set and distance function.

- **min_samples**: The number of samples (or total weight) in a neighborhood for a point  to be considered as a core point. This includes the point itself. If `min_samples`  is set to a higher value, DBSCAN will find denser clusters, whereas if it is set  to a lower value, the found clusters will be more sparse.

- **metric**: The metric to use when calculating distance between instances in a feature  array. If metric is a string or callable, it must be one of the options allowed  by `sklearn.metrics.pairwise_distances` for its metric parameter.

- **n_jobs**: The number of parallel jobs to run.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[projection]
		end
		subgraph New columns
			OCOL0[cluster_label]
		end
	end

	subgraph DBSCAN
		StepInput[Input Columns: projection]
		StepOutput[Output Columns: cluster_label]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **projection** (`List[float]`): Vector representation of the text to cluster,  normally the output from the `UMAP` step.




#### Outputs


- **cluster_label** (`int`): Integer representing the label of a given cluster. -1  means it wasn't clustered.







### References

- [DBSCAN demo of sklearn](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#demo-of-dbscan-clustering-algorithm)

- [sklearn dbscan](https://scikit-learn.org/stable/modules/clustering.html#dbscan)


