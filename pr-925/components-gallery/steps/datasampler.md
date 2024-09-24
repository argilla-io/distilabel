---
hide:
  - navigation
---
# DataSampler

Step to sample from a dataset.



`GeneratorStep` that samples from a dataset and yields it in batches.





### Attributes

- **data**: The list of dictionaries to sample from.

- **size**: The number of samples per example.

- **samples**: The number of examples.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph New columns
			OCOL0[dynamic]
		end
	end

	subgraph DataSampler
		StepOutput[Output Columns: dynamic]
	end

	StepOutput --> OCOL0

```




#### Outputs


- **dynamic** (based on the keys found on the first dictionary of the list): The columns  of the dataset.





### Examples


#### Sample data from a list of dictionaries
```python
from distilabel.steps import DataSampler

sampler = DataSampler(
    data=[{"sample": f"sample {i}"} for i in range(30)],
    samples=10,
    size=2,
    batch_size=4
)
sampler.load()

result = next(sampler.process())
# >>> result
# ([{'sample': ['sample 7', 'sample 0']}, {'sample': ['sample 2', 'sample 21']}, {'sample': ['sample 17', 'sample 12']}, {'sample': ['sample 2', 'sample 14']}], False)
```

#### Pipeline with a loader and a sampler combined in a single stream
```python
from datasets import load_dataset

from distilabel.steps import LoadDataFromDicts, DataSampler
from distilabel.steps.tasks.apigen.utils import PrepareExamples
from distilabel.pipeline import Pipeline

ds = (
    load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    .shuffle(seed=42)
    .select(range(500))
    .to_list()
)

with Pipeline(name="APIGenPipeline") as pipeline:
    loader_seeds = LoadDataFromDicts(data=data)
    sampler = DataSampler(
        data=ds,
        size=2,
        samples=len(data),
        batch_size=8,
    )
    prep_examples = PrepareExamples()

    sampler >> prep_examples
    (
        [loader_seeds, prep_examples]
        >> combine_steps
    )
# Now we have a single stream of data with the loader and the sampler data
```




