---
hide:
  - navigation
---
# TextClustering

Task that clusters a set of texts and generates summary labels for each cluster.



This is a `GlobalTask` that inherits from `TextClassification`, this means that all
    the attributes from that class are available here. Also, in this case we deal
    with all the inputs at once, instead of using batches. The `input_batch_size` is
    used here to send the examples to the LLM in batches (a subtle difference with the
    more common `Task` definitions).
    The task looks in each cluster for a given number of representative examples (the number
    is set by the `samples_per_cluster` attribute), and sends them to the LLM to get a label/s
    that represent the cluster. The labels are then assigned to each text in the cluster.
    The clusters and projections used in the step, are assumed to be obtained from the `UMAP`
    + `DBSCAN` steps, but could be generated for similar steps, as long as they represent the
    same concepts.
    This step runs a pipeline like the one in this repository:
    https://github.com/huggingface/text-clustering





### Attributes

- **savefig**: Whether to generate and save a figure with the clustering of the texts.  - samples_per_cluster: The number of examples to use in the LLM as a sample of the cluster.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[text]
			ICOL1[projection]
			ICOL2[cluster_label]
		end
		subgraph New columns
			OCOL0[summary_label]
			OCOL1[model_name]
		end
	end

	subgraph TextClustering
		StepInput[Input Columns: text, projection, cluster_label]
		StepOutput[Output Columns: summary_label, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **text** (`str`): The reference text we want to obtain labels for.

- **projection** (`List[float]`): Vector representation of the text to cluster,  normally the output from the `UMAP` step.

- **cluster_label** (`int`): Integer representing the label of a given cluster. -1  means it wasn't clustered.




#### Outputs


- **summary_label** (`str`): The label or list of labels for the text.

- **model_name** (`str`): The name of the model used to generate the label/s.





### Examples


#### Generate labels for a set of texts using clustering
```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps import UMAP, DBSCAN, TextClustering
from distilabel.pipeline import Pipeline

ds_name = "argilla-warehouse/personahub-fineweb-edu-4-clustering-100k"

with Pipeline(name="Text clustering dataset") as pipeline:
    batch_size = 500

    ds = load_dataset(ds_name, split="train").select(range(10000))
    loader = make_generator_step(ds, batch_size=batch_size, repo_id=ds_name)

    umap = UMAP(n_components=2, metric="cosine")
    dbscan = DBSCAN(eps=0.3, min_samples=30)

    text_clustering = TextClustering(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
        n=3,  # 3 labels per example
        query_title="Examples of Personas",
        samples_per_cluster=10,
        context=(
            "Describe the main themes, topics, or categories that could describe the "
            "following types of personas. All the examples of personas must share "
            "the same set of labels."
        ),
        default_label="None",
        savefig=True,
        input_batch_size=8,
        input_mappings={"text": "persona"},
        use_default_structured_output=True,
    )

    loader >> umap >> dbscan >> text_clustering
```




### References

- [text-clustering repository](https://github.com/huggingface/text-clustering)


