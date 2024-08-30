---
hide:
  - navigation
---
# CombineOutputs

Combine the outputs of several upstream steps.



`CombineOutputs` is a `Step` that takes the outputs of several upstream steps and combines
    them to generate a new dictionary with all keys/columns of the upstream steps outputs.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[dynamic]
		end
		subgraph New columns
			OCOL0[dynamic]
		end
	end

	subgraph CombineOutputs
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (based on the upstream `Step`s): All the columns of the upstream steps outputs.




#### Outputs


- **dynamic** (based on the upstream `Step`s): All the columns of the upstream steps outputs.





### Examples


#### Combine dictionaries of a dataset
```python
from distilabel.steps import CombineOutputs

combine_outputs = CombineOutputs()
combine_outputs.load()

result = next(
    combine_outputs.process(
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        [{"c": 5, "d": 6}, {"c": 7, "d": 8}],
    )
)
# [
#   {"a": 1, "b": 2, "c": 5, "d": 6},
#   {"a": 3, "b": 4, "c": 7, "d": 8},
# ]
```

#### Combine upstream steps outputs in a pipeline
```python
from distilabel.pipeline import Pipeline
from distilabel.steps import CombineOutputs

with Pipeline() as pipeline:
    step_1 = ...
    step_2 = ...
    step_3 = ...
    combine = CombineOutputs()

    [step_1, step_2, step_3] >> combine
```




