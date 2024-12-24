---
hide:
  - navigation
---
# KeepColumns

Keeps selected columns in the dataset.



`KeepColumns` is a `Step` that implements the `process` method that keeps only the columns
    specified in the `columns` attribute. Also `KeepColumns` provides an attribute `columns` to
    specify the columns to keep which will override the default value for the properties `inputs`
    and `outputs`.



### Note
The order in which the columns are provided is important, as the output will be sorted
using the provided order, which is useful before pushing either a `dataset.Dataset` via
the `PushToHub` step or a `distilabel.Distiset` via the `Pipeline.run` output variable.



### Attributes

- **columns**: List of strings with the names of the columns to keep.





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

	subgraph KeepColumns
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (determined by `columns` attribute): The columns to keep.




#### Outputs


- **dynamic** (determined by `columns` attribute): The columns that were kept.





### Examples


#### Select the columns to keep
```python
from distilabel.steps import KeepColumns

keep_columns = KeepColumns(
    columns=["instruction", "generation"],
)
keep_columns.load()

result = next(
    keep_columns.process(
        [{"instruction": "What's the brightest color?", "generation": "white", "model_name": "my_model"}],
    )
)
# >>> result
# [{'instruction': "What's the brightest color?", 'generation': 'white'}]
```




