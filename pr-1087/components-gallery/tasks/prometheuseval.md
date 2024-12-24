---
hide:
  - navigation
---
# PrometheusEval

Critique and rank the quality of generations from an `LLM` using Prometheus 2.0.



`PrometheusEval` is a task created for Prometheus 2.0, covering both the absolute and relative
    evaluations. The absolute evaluation i.e. `mode="absolute"` is used to evaluate a single generation from
    an LLM for a given instruction. The relative evaluation i.e. `mode="relative"` is used to evaluate two generations from an LLM
    for a given instruction.
    Both evaluations provide the possibility of using a reference answer to compare with or withoug
    the `reference` attribute, and both are based on a score rubric that critiques the generation/s
    based on the following default aspects: `helpfulness`, `harmlessness`, `honesty`, `factual-validity`,
    and `reasoning`, that can be overridden via `rubrics`, and the selected rubric is set via the attribute
    `rubric`.



### Note
The `PrometheusEval` task is better suited and intended to be used with any of the Prometheus 2.0
models released by Kaist AI, being: https://huggingface.co/prometheus-eval/prometheus-7b-v2.0,
and https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0. The critique assessment formatting
and quality is not guaranteed if using another model, even though some other models may be able to
correctly follow the formatting and generate insightful critiques too.



### Attributes

- **mode**: the evaluation mode to use, either `absolute` or `relative`. It defines whether the task  will evaluate one or two generations.

- **rubric**: the score rubric to use within the prompt to run the critique based on different aspects.  Can be any existing key in the `rubrics` attribute, which by default means that it can be:  `helpfulness`, `harmlessness`, `honesty`, `factual-validity`, or `reasoning`. Those will only  work if using the default `rubrics`, otherwise, the provided `rubrics` should be used.

- **rubrics**: a dictionary containing the different rubrics to use for the critique, where the keys are  the rubric names and the values are the rubric descriptions. The default rubrics are the following:  `helpfulness`, `harmlessness`, `honesty`, `factual-validity`, and `reasoning`.

- **reference**: a boolean flag to indicate whether a reference answer / completion will be provided, so  that the model critique is based on the comparison with it. It implies that the column `reference`  needs to be provided within the input data in addition to the rest of the inputs.

- **system_prompt**: The system prompt for the PrometheusEval task.

- **_template**: a Jinja2 template used to format the input for the LLM.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[generation]
			ICOL2[generations]
			ICOL3[reference]
			ICOL4[system_prompt]
		end
		subgraph New columns
			OCOL0[feedback]
			OCOL1[result]
			OCOL2[model_name]
		end
	end

	subgraph PrometheusEval
		StepInput[Input Columns: instruction, generation, generations, reference, system_prompt]
		StepOutput[Output Columns: feedback, result, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	ICOL3 --> StepInput
	ICOL4 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction to use as reference.

- **generation** (`str`, optional): The generated text from the given `instruction`. This column is required  if `mode=absolute`.

- **generations** (`List[str]`, optional): The generated texts from the given `instruction`. It should  contain 2 generations only. This column is required if `mode=relative`.

- **reference** (`str`, optional): The reference / golden answer for the `instruction`, to be used by the LLM  for comparison against.

- **system_prompt** (`Optional[str]`): The system prompt for the PrometheusEval task.




#### Outputs


- **feedback** (`str`): The feedback explaining the result below, as critiqued by the LLM using the  pre-defined score rubric, compared against `reference` if provided.

- **result** (`Union[int, Literal["A", "B"]]`): If `mode=absolute`, then the result contains the score for the  `generation` in a likert-scale from 1-5, otherwise, if `mode=relative`, then the result contains either  "A" or "B", the "winning" one being the generation in the index 0 of `generations` if `result='A'` or the  index 1 if `result='B'`.

- **model_name** (`str`): The model name used to generate the `feedback` and `result`.





### Examples


#### Critique and evaluate LLM generation quality using Prometheus 2_0
```python
from distilabel.steps.tasks import PrometheusEval
from distilabel.models import vLLM

# Consider this as a placeholder for your actual LLM.
prometheus = PrometheusEval(
    llm=vLLM(
        model="prometheus-eval/prometheus-7b-v2.0",
        chat_template="[INST] {{ messages[0]"content" }}\n{{ messages[1]"content" }}[/INST]",
    ),
    mode="absolute",
    rubric="factual-validity"
)

prometheus.load()

result = next(
    prometheus.process(
        [
            {"instruction": "make something", "generation": "something done"},
        ]
    )
)
# result
# [
#     {
#         'instruction': 'make something',
#         'generation': 'something done',
#         'model_name': 'prometheus-eval/prometheus-7b-v2.0',
#         'feedback': 'the feedback',
#         'result': 6,
#     }
# ]
```

#### Critique for relative evaluation
```python
from distilabel.steps.tasks import PrometheusEval
from distilabel.models import vLLM

# Consider this as a placeholder for your actual LLM.
prometheus = PrometheusEval(
    llm=vLLM(
        model="prometheus-eval/prometheus-7b-v2.0",
        chat_template="[INST] {{ messages[0]"content" }}\n{{ messages[1]"content" }}[/INST]",
    ),
    mode="relative",
    rubric="honesty"
)

prometheus.load()

result = next(
    prometheus.process(
        [
            {"instruction": "make something", "generations": ["something done", "other thing"]},
        ]
    )
)
# result
# [
#     {
#         'instruction': 'make something',
#         'generations': ['something done', 'other thing'],
#         'model_name': 'prometheus-eval/prometheus-7b-v2.0',
#         'feedback': 'the feedback',
#         'result': 'something done',
#     }
# ]
```

#### Critique with a custom rubric
```python
from distilabel.steps.tasks import PrometheusEval
from distilabel.models import vLLM

# Consider this as a placeholder for your actual LLM.
prometheus = PrometheusEval(
    llm=vLLM(
        model="prometheus-eval/prometheus-7b-v2.0",
        chat_template="[INST] {{ messages[0]"content" }}\n{{ messages[1]"content" }}[/INST]",
    ),
    mode="absolute",
    rubric="custom",
    rubrics={
        "custom": "[A]\nScore 1: A\nScore 2: B\nScore 3: C\nScore 4: D\nScore 5: E"
    }
)

prometheus.load()

result = next(
    prometheus.process(
        [
            {"instruction": "make something", "generation": "something done"},
        ]
    )
)
# result
# [
#     {
#         'instruction': 'make something',
#         'generation': 'something done',
#         'model_name': 'prometheus-eval/prometheus-7b-v2.0',
#         'feedback': 'the feedback',
#         'result': 6,
#     }
# ]
```

#### Critique using a reference answer
```python
from distilabel.steps.tasks import PrometheusEval
from distilabel.models import vLLM

# Consider this as a placeholder for your actual LLM.
prometheus = PrometheusEval(
    llm=vLLM(
        model="prometheus-eval/prometheus-7b-v2.0",
        chat_template="[INST] {{ messages[0]"content" }}\n{{ messages[1]"content" }}[/INST]",
    ),
    mode="absolute",
    rubric="helpfulness",
    reference=True,
)

prometheus.load()

result = next(
    prometheus.process(
        [
            {
                "instruction": "make something",
                "generation": "something done",
                "reference": "this is a reference answer",
            },
        ]
    )
)
# result
# [
#     {
#         'instruction': 'make something',
#         'generation': 'something done',
#         'reference': 'this is a reference answer',
#         'model_name': 'prometheus-eval/prometheus-7b-v2.0',
#         'feedback': 'the feedback',
#         'result': 6,
#     }
# ]
```




### References

- [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://arxiv.org/abs/2405.01535)

- [prometheus-eval: Evaluate your LLM's response with Prometheus 💯](https://github.com/prometheus-eval/prometheus-eval)


