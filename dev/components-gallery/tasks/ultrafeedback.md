# UltraFeedback


Rank generations focusing on different aspects using an `LLM`.



UltraFeedback: Boosting Language Models with High-quality Feedback.





### Attributes

- **aspect**: The aspect to perform with the `UltraFeedback` model. The available aspects are:  - `helpfulness`: Evaluate text outputs based on helpfulness.  - `honesty`: Evaluate text outputs based on honesty.  - `instruction-following`: Evaluate text outputs based on given instructions.  - `truthfulness`: Evaluate text outputs based on truthfulness.  Additionally, a custom aspect has been defined by Argilla, so as to evaluate the overall  assessment of the text outputs within a single prompt. The custom aspect is:  - `overall-rating`: Evaluate text outputs based on an overall assessment.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[generations]
		end
		subgraph New columns
			OCOL0[ratings]
			OCOL1[rationales]
			OCOL2[model_name]
		end
	end

	subgraph UltraFeedback
		StepInput[Input Columns: instruction, generations]
		StepOutput[Output Columns: ratings, rationales, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The reference instruction to evaluate the text outputs.

- **generations** (`List[str]`): The text outputs to evaluate for the given instruction.




#### Outputs


- **ratings** (`List[float]`): The ratings for each of the provided text outputs.

- **rationales** (`List[str]`): The rationales for each of the provided text outputs.

- **model_name** (`str`): The name of the model used to generate the ratings and rationales.





### Examples


#### Rate generations from different LLMs based on the selected aspect
```python
from distilabel.steps.tasks import UltraFeedback
from distilabel.llms.huggingface import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
ultrafeedback = UltraFeedback(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    )
)

ultrafeedback.load()

result = next(
    chat.process(
        [
            {
                "instruction": "How much is 2+2?",
                "generations": ["4", "and a car"],
            }
        ]
    )
)
# result
# [
#     {
#         'instruction': 'How much is 2+2?',
#         'generations': ['4', 'and a car'],
#         'ratings': [1, 2],
#         'rationales': ['explanation for 4', 'explanation for and a car'],
#         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
#     }
# ]
```




### References

- [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377)

- [UltraFeedback - GitHub Repository](https://github.com/OpenBMB/UltraFeedback)


