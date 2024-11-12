---
hide:
  - navigation
---
# ExpandColumns

Expand columns that contain lists into multiple rows.



`ExpandColumns` is a `Step` that takes a list of columns and expands them into multiple
    rows. The new rows will have the same data as the original row, except for the expanded
    column, which will contain a single item from the original list.





### Attributes

- **columns**: A dictionary that maps the column to be expanded to the new column name  or a list of columns to be expanded. If a list is provided, the new column name  will be the same as the column name.

- **encoded**: A bool to inform Whether the columns are JSON encoded lists. If this value is  set to True, the columns will be decoded before expanding. Alternatively, to specify  columns that can be encoded, a list can be provided. In this case, the column names  informed must be a subset of the columns selected for expansion.





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

	subgraph ExpandColumns
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (determined by `columns` attribute): The columns to be expanded into  multiple rows.




#### Outputs


- **dynamic** (determined by `columns` attribute): The expanded columns.





### Examples


#### Expand the selected columns into multiple rows
```python
from distilabel.steps import ExpandColumns

expand_columns = ExpandColumns(
    columns=["generation"],
)
expand_columns.load()

result = next(
    expand_columns.process(
        [
            {
                "instruction": "instruction 1",
                "generation": ["generation 1", "generation 2"]}
        ],
    )
)
# >>> result
# [{'instruction': 'instruction 1', 'generation': 'generation 1'}, {'instruction': 'instruction 1', 'generation': 'generation 2'}]
```

#### Expand the selected columns which are JSON encoded into multiple rows
```python
from distilabel.steps import ExpandColumns

expand_columns = ExpandColumns(
    columns=["generation"],
    encoded=True,  # It can also be a list of columns that are encoded, i.e. ["generation"]
)
expand_columns.load()

result = next(
    expand_columns.process(
        [
            {
                "instruction": "instruction 1",
                "generation": '["generation 1", "generation 2"]'}
        ],
    )
)
# >>> result
# [{'instruction': 'instruction 1', 'generation': 'generation 1'}, {'instruction': 'instruction 1', 'generation': 'generation 2'}]
```




