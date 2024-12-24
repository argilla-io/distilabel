---
hide:
  - navigation
---
# ComplexityScorer

Score instructions based on their complexity using an `LLM`.



`ComplexityScorer` is a pre-defined task used to rank a list of instructions based in
    their complexity. It's an implementation of the complexity score task from the paper
    'What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection
    in Instruction Tuning'.





### Attributes

- **system_prompt**: The system prompt for the complexity scorer task.

- **_template**: a Jinja2 template used to format the input for the LLM.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instructions]
			ICOL1[system_prompt]
		end
		subgraph New columns
			OCOL0[scores]
			OCOL1[model_name]
		end
	end

	subgraph ComplexityScorer
		StepInput[Input Columns: instructions, system_prompt]
		StepOutput[Output Columns: scores, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **instructions** (`List[str]`): The list of instructions to be scored.

- **system_prompt** (`Optional[str]`): The system prompt for the complexity scorer task.




#### Outputs


- **scores** (`List[float]`): The score for each instruction.

- **model_name** (`str`): The model name used to generate the scores.





### Examples


#### Evaluate the complexity of your instructions
```python
from distilabel.steps.tasks import ComplexityScorer
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
scorer = ComplexityScorer(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    )
)

scorer.load()

result = next(
    scorer.process(
        [{"instructions": ["plain instruction", "highly complex instruction"]}]
    )
)
# result
# [{'instructions': ['plain instruction', 'highly complex instruction'], 'model_name': 'test', 'scores': [1, 5], 'distilabel_metadata': {'raw_output_complexity_scorer_0': 'output'}}]
```

#### Generate structured output with default schema
```python
from distilabel.steps.tasks import ComplexityScorer
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
scorer = ComplexityScorer(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    use_default_structured_output=use_default_structured_output
)

scorer.load()

result = next(
    scorer.process(
        [{"instructions": ["plain instruction", "highly complex instruction"]}]
    )
)
# result
# [{'instructions': ['plain instruction', 'highly complex instruction'], 'model_name': 'test', 'scores': [1, 2], 'distilabel_metadata': {'raw_output_complexity_scorer_0': '{ \n  "scores": [\n    1, \n    2\n  ]\n}'}}]
```




### References

- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)


