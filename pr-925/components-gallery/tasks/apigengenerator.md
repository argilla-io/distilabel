---
hide:
  - navigation
---
# APIGenGenerator

Generate queries and answers for the given functions in JSON format.



The `APIGenGenerator` is inspired by the APIGen pipeline, which was designed to generate
    verifiable and diverse function-calling datasets. The task generates a set of diverse queries
    and corresponding answers for the given functions in JSON format.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[examples]
			ICOL1[func_name]
			ICOL2[func_desc]
		end
		subgraph New columns
			OCOL0[queries]
			OCOL1[answers]
		end
	end

	subgraph APIGenGenerator
		StepInput[Input Columns: examples, func_name, func_desc]
		StepOutput[Output Columns: queries, answers]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **examples** (`str`): Examples used as few shots to guide the model.

- **func_name** (`str`): Name for the function to generate.

- **func_desc** (`str`): Description of what the function should do.




#### Outputs


- **queries** (`List[str]`): The list of queries.

- **answers** (`List[str]`): The list of answers.





### Examples


#### Generate without structured output (original implementation)
```python
from distilabel.steps.tasks import ApiGenGenerator
from distilabel.llms import InferenceEndpointsLLM

llm=InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 1024,
    },
)
apigen = ApiGenGenerator(
    use_default_structured_output=False,
    llm=llm
)
apigen.load()

res = next(
    apigen.process(
        [
            {
                "examples": 'QUERY:
What is the binary sum of 10010 and 11101?
ANSWER:
[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
                "func_name": "getrandommovie",
                "func_desc": "Returns a list of random movies from a database by calling an external API."
            }
        ]
    )
)
res
# [{'examples': 'QUERY:
What is the binary sum of 10010 and 11101?
ANSWER:
[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
# 'number': 1,
# 'func_name': 'getrandommovie',
# 'func_desc': 'Returns a list of random movies from a database by calling an external API.',
# 'queries': ['I want to watch a movie tonight, can you recommend a random one from your database?',
# 'Give me 5 random movie suggestions from your database to plan my weekend.'],
# 'answers': [[{'name': 'getrandommovie', 'arguments': {}}],
# [{'name': 'getrandommovie', 'arguments': {}},
#     {'name': 'getrandommovie', 'arguments': {}},
#     {'name': 'getrandommovie', 'arguments': {}},
#     {'name': 'getrandommovie', 'arguments': {}},
#     {'name': 'getrandommovie', 'arguments': {}}]],
# 'distilabel_metadata': {'raw_output_api_gen_generator_0': '[
   {
       "query": "I want to watch a movie tonight, can you recommend a random one from your database?",
       "answers": [
   {
       "name": "getrandommovie",
       "arguments": {}
   }
       ]
   },
   {
       "query": "Give me 5 random movie suggestions from your database to plan my weekend.",
       "answers": [
   {
       "name": "getrandommovie",
       "arguments": {}
   },
   {
       "name": "getrandommovie",
       "arguments": {}
   },
   {
       "name": "getrandommovie",
       "arguments": {}
   },
   {
       "name": "getrandommovie",
       "arguments": {}
   },
   {
       "name": "getrandommovie",
       "arguments": {}
   }
       ]
   }
]',
# 'raw_input_api_gen_generator_0': [{'role': 'system',
#     'content': "You are a data labeler. Your responsibility is to generate a set of diverse queries and corresponding answers for the given functions in JSON format.

Construct queries and answers that exemplify how to use these functions in a practical scenario. Include in each query specific, plausible values for each parameter. For instance, if the function requires a date, use a typical and reasonable date.

Ensure the query:
- Is clear and concise
- Demonstrates typical use cases
- Includes all necessary parameters in a meaningful way. For numerical parameters, it could be either numbers or words
- Across a variety level of difficulties, ranging from beginner and advanced use cases
- The corresponding result's parameter types and ranges match with the function's descriptions

Ensure the answer:
- Is a list of function calls in JSON format
- The length of the answer list should be equal to the number of requests in the query
- Can solve all the requests in the query effectively"},
#     {'role': 'user',
#     'content': 'Here are examples of queries and the corresponding answers for similar functions:
QUERY:
What is the binary sum of 10010 and 11101?
ANSWER:
[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]

Note that the query could be interpreted as a combination of several independent requests.
Based on these examples, generate 2 diverse query and answer pairs for the function `getrandommovie`
The detailed function description is the following:
Returns a list of random movies from a database by calling an external API.

The output MUST strictly adhere to the following JSON format, and NO other text MUST be included:
```

#### Generate with structured output
```python
from distilabel.steps.tasks import ApiGenGenerator
from distilabel.llms import InferenceEndpointsLLM

llm=InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tokenizer="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 1024,
    },
)
apigen = ApiGenGenerator(
    use_default_structured_output=True,
    llm=llm
)
apigen.load()

res_struct = next(
    apigen.process(
        [
            {
                "examples": 'QUERY:
What is the binary sum of 10010 and 11101?
ANSWER:
[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
                "func_name": "getrandommovie",
                "func_desc": "Returns a list of random movies from a database by calling an external API."
            }
        ]
    )
)
res_struct
# [{'examples': 'QUERY:
What is the binary sum of 10010 and 11101?
ANSWER:
[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
# 'number': 1,
# 'func_name': 'getrandommovie',
# 'func_desc': 'Returns a list of random movies from a database by calling an external API.',
# 'queries': ["I'm bored and want to watch a movie. Can you suggest some movies?",
# "My family and I are planning a movie night. We can't decide on what to watch. Can you suggest some random movie titles?"],
# 'answers': [[{'arguments': {}, 'name': 'getrandommovie'}],
# [{'arguments': {}, 'name': 'getrandommovie'}]],
# 'distilabel_metadata': {'raw_output_api_gen_generator_0': '{ 
  "pairs": [
    {
      "answers": [
{
  "arguments": {},
  "name": "getrandommovie"
}
      ],
      "query": "I'm bored and want to watch a movie. Can you suggest some movies?"
    },
    {
      "answers": [
{
  "arguments": {},
  "name": "getrandommovie"
}
      ],
      "query": "My family and I are planning a movie night. We can't decide on what to watch. Can you suggest some random movie titles?"
    }
  ]
}',
# 'raw_input_api_gen_generator_0': [{'role': 'system',
#     'content': "You are a data labeler. Your responsibility is to generate a set of diverse queries and corresponding answers for the given functions in JSON format.

Construct queries and answers that exemplify how to use these functions in a practical scenario. Include in each query specific, plausible values for each parameter. For instance, if the function requires a date, use a typical and reasonable date.

Ensure the query:
- Is clear and concise
- Demonstrates typical use cases
- Includes all necessary parameters in a meaningful way. For numerical parameters, it could be either numbers or words
- Across a variety level of difficulties, ranging from beginner and advanced use cases
- The corresponding result's parameter types and ranges match with the function's descriptions

Ensure the answer:
- Is a list of function calls in JSON format
- The length of the answer list should be equal to the number of requests in the query
- Can solve all the requests in the query effectively"},
#     {'role': 'user',
#     'content': 'Here are examples of queries and the corresponding answers for similar functions:
QUERY:
What is the binary sum of 10010 and 11101?
ANSWER:
[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]

Note that the query could be interpreted as a combination of several independent requests.
Based on these examples, generate 2 diverse query and answer pairs for the function `getrandommovie`
The detailed function description is the following:
Returns a list of random movies from a database by calling an external API.

Now please generate 2 diverse query and answer pairs following the above format.'}]},
# 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
```




### References

- [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)

- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)


