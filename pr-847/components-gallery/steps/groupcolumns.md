---
hide:
  - navigation
---
# GroupColumns

Combines columns from a list of `StepInput`.



`GroupColumns` is a `Step` that implements the `process` method that calls the `group_dicts`
    function to handle and combine a list of `StepInput`. Also `GroupColumns` provides two attributes
    `columns` and `output_columns` to specify the columns to group and the output columns
    which will override the default value for the properties `inputs` and `outputs`, respectively.





### Attributes

- **columns**: List of strings with the names of the columns to group.

- **output_columns**: Optional list of strings with the names of the output columns.





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

	subgraph GroupColumns
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (determined by `columns` attribute): The columns to group.




#### Outputs


- **dynamic** (determined by `columns` and `output_columns` attributes): The columns  that were grouped.





### Examples


#### Combine columns of a dataset
```python
from distilabel.steps import GroupColumns

group_columns = GroupColumns(
    name="group_columns",
    columns=["generation", "model_name"],
)
group_columns.load()

result = next(
    group_columns.process(
        [{"generation": "AI generated text"}, {"model_name": "my_model"}],
        [{"generation": "Other generated text", "model_name": "my_model"}]
    )
)
# >>> result
# [{'merged_generation': ['AI generated text', 'Other generated text'], 'merged_model_name': ['my_model']}]
```

#### Specify the name of the output columns
```python
from distilabel.steps import GroupColumns

group_columns = GroupColumns(
    name="group_columns",
    columns=["generation", "model_name"],
    output_columns=["generations", "generation_models"]
)
group_columns.load()

result = next(
    group_columns.process(
        [{"generation": "AI generated text"}, {"model_name": "my_model"}],
        [{"generation": "Other generated text", "model_name": "my_model"}]
    )
)
# >>> result
#[{'generations': ['AI generated text', 'Other generated text'], 'generation_models': ['my_model']}]
```




