---
hide:
  - navigation
---
# PrepareExamples

Helper step to create examples from `query` and `answers` pairs used as Few Shots in APIGen.










### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[query]
			ICOL1[answers]
		end
		subgraph New columns
			OCOL0[examples]
		end
	end

	subgraph PrepareExamples
		StepInput[Input Columns: query, answers]
		StepOutput[Output Columns: examples]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **query** (`str`): The query to generate examples from.

- **answers** (`str`): The answers to the query.




#### Outputs


- **examples** (`str`): The formatted examples.





### Examples


#### Generate examples for APIGen
```python
from distilabel.steps.tasks.apigen.utils import PrepareExamples

prepare_examples = PrepareExamples()
result = next(prepare_examples.process(
    [
        {
            "query": ['I need the area of circles with radius 2.5, 5, and 7.5 inches, please.', 'Can you provide the current locations of buses and trolleys on route 12?'],
            "answers": ['[{"name": "circle_area", "arguments": {"radius": 2.5}}, {"name": "circle_area", "arguments": {"radius": 5}}, {"name": "circle_area", "arguments": {"radius": 7.5}}]', '[{"name": "bus_trolley_locations", "arguments": {"route": "12"}}]']
        }
    ]
)
# result
# [{'examples': '## Query:\nI need the area of circles with radius 2.5, 5, and 7.5 inches, please.\n## Answers:\n[{"name": "circle_area", "arguments": {"radius": 2.5}}, {"name": "circle_area", "arguments": {"radius": 5}}, {"name": "circle_area", "arguments": {"radius": 7.5}}]\n\n## Query:\nCan you provide the current locations of buses and trolleys on route 12?\n## Answers:\n[{"name": "bus_trolley_locations", "arguments": {"route": "12"}}]'}, {'examples': '## Query:\nI need the area of circles with radius 2.5, 5, and 7.5 inches, please.\n## Answers:\n[{"name": "circle_area", "arguments": {"radius": 2.5}}, {"name": "circle_area", "arguments": {"radius": 5}}, {"name": "circle_area", "arguments": {"radius": 7.5}}]\n\n## Query:\nCan you provide the current locations of buses and trolleys on route 12?\n## Answers:\n[{"name": "bus_trolley_locations", "arguments": {"route": "12"}}]'}]
```




