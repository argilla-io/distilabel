---
hide:
  - navigation
---
# QualityScorer

Score responses based on their quality using an `LLM`.



`QualityScorer` is a pre-defined task that defines the `instruction` as the input
    and `score` as the output. This task is used to rate the quality of instructions and responses.
    It's an implementation of the quality score task from the paper 'What Makes Good Data
    for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning'.
    The task follows the same scheme as the Complexity Scorer, but the instruction-response pairs
    are scored in terms of quality, obtaining a quality score for each instruction.





### Attributes

- **system_prompt**: The system prompt for the quality scorer task.

- **_template**: a Jinja2 template used to format the input for the LLM.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[responses]
			ICOL2[system_prompt]
		end
		subgraph New columns
			OCOL0[scores]
			OCOL1[model_name]
		end
	end

	subgraph QualityScorer
		StepInput[Input Columns: instruction, responses, system_prompt]
		StepOutput[Output Columns: scores, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction that was used to generate the `responses`.

- **responses** (`List[str]`): The responses to be scored. Each response forms a pair with the instruction.

- **system_prompt** (`Optional[str]`): The system prompt for the quality scorer task.




#### Outputs


- **scores** (`List[float]`): The score for each instruction.

- **model_name** (`str`): The model name used to generate the scores.





### Examples


#### Evaluate the quality of your instructions
```python
from distilabel.steps.tasks import QualityScorer
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
scorer = QualityScorer(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    )
)

scorer.load()

result = next(
    scorer.process(
        [
            {
                "instruction": "instruction",
                "responses": ["good response", "weird response", "bad response"]
            }
        ]
    )
)
# result
[
    {
        'instructions': 'instruction',
        'model_name': 'test',
        'scores': [5, 3, 1],
    }
]
```

#### Generate structured output with default schema
```python
from distilabel.steps.tasks import QualityScorer
from distilabel.models import InferenceEndpointsLLM

scorer = QualityScorer(
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    use_default_structured_output=True
)

scorer.load()

result = next(
    scorer.process(
        [
            {
                "instruction": "instruction",
                "responses": ["good response", "weird response", "bad response"]
            }
        ]
    )
)

# result
[{'instruction': 'instruction',
'responses': ['good response', 'weird response', 'bad response'],
'scores': [1, 2, 3],
'distilabel_metadata': {'raw_output_quality_scorer_0': '{  "scores": [1, 2, 3] }'},
'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
```




### References

- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)


