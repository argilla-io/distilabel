# CombineColumns


Combines columns from a list of `StepInput`.



`CombineColumns` is a `Step` that implements the `process` method that calls the `combine_dicts`
    function to handle and combine a list of `StepInput`. Also `CombineColumns` provides two attributes
    `columns` and `output_columns` to specify the columns to merge and the output columns
    which will override the default value for the properties `inputs` and `outputs`, respectively.





### Attributes

- **columns**: List of strings with the names of the columns to merge.

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

	subgraph CombineColumns
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (determined by `columns` attribute): The columns to merge.




#### Outputs


- **dynamic** (determined by `columns` and `output_columns` attributes): The columns  that were merged.





### Examples


#### Combine columns of a dataset
```python
from distilabel.steps import CombineColumns

combine_columns = CombineColumns(
    name="combine_columns",
    columns=["generation", "model_name"],
)
combine_columns.load()

result = next(
    combine_columns.process(
        [{"generation": "AI generated text"}, {"model_name": "my_model"}],
        [{"generation": "Other generated text", "model_name": "my_model"}]
    )
)
# >>> result
# [{'merged_generation': ['AI generated text', 'Other generated text'], 'merged_model_name': ['my_model']}]
```

#### Specify the name of the output columns
```python
from distilabel.steps import CombineColumns

combine_columns = CombineColumns(
    name="combine_columns",
    columns=["generation", "model_name"],
    output_columns=["generations", "generation_models"]
)
combine_columns.load()

result = next(
    combine_columns.process(
        [{"generation": "AI generated text"}, {"model_name": "my_model"}],
        [{"generation": "Other generated text", "model_name": "my_model"}]
    )
)
# >>> result
#[{'generations': ['AI generated text', 'Other generated text'], 'generation_models': ['my_model']}]
```




