# LoadDataFromDicts


Loads a dataset from a list of dictionaries.



`GeneratorStep` that loads a dataset from a list of dictionaries and yields it in
    batches.



### Attributes

- **data**: The list of dictionaries to load the data from.




### Runtime Parameters

- **batch_size**: The batch size to use when processing the data.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph New columns
			OCOL0[dynamic]
		end
	end

	subgraph LoadDataFromDicts
		StepOutput[Output Columns: dynamic]
	end

	StepOutput --> OCOL0

```




#### Outputs


- **dynamic** (based on the keys found on the first dictionary of the list): The columns  of the dataset.







