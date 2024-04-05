# Feedback Tasks

This section presents tasks that work on the `LLM` output to return some feedback on the inputs, like `UltraFeedback` which asks an `LLM` to rate the responses according to some criteria.

Include here tasks like `UltraFeedback` that are intended to generate some feedback (in that case for preference) over some outputs

## UltraFeedback

[`UltraFeedback`][distilabel.steps.tasks.ultrafeedback] is a `Task` inspired from [`UltraFeedback: Boosting Language Models with High-quality Feedback`](https://arxiv.org/abs/2310.01377), where the authors present the methodology that leaded to the creation of their famous dataset:

```python
from distilabel.steps.tasks import UltraFeedback
from distilabel.llms import OpenAILLM

ultrafeedback = UltraFeedback(
    name="ultrafeedback_openai",
    llm=OpenAILLM(model="gpt-4"),
    aspect="overall-rating",
)
```

Will see in the following section what this `aspect` does, let's take a look first at how this task works:

```python
result = ultrafeedback.process(
    [
        {
            "instruction": "Arianna has 12 chocolates more than Danny. Danny has 6 chocolates more than Robbie. Arianna has twice as many chocolates as Robbie has. How many chocolates does Danny have?",
            "generations": [
                "To find the number of chocolates Danny has, we can first write an equation that relates the number of chocolates Arianna, Danny, and Robbie have.\n\nArianna has 12 more chocolates than Danny, so we can write:\n\nArianna = Danny + 12 (1)\n\nDanny has 6 more chocolates than Robbie, so we can write:\n\nDanny = Robbie + 6 (2)\n\nArianna has twice as many chocolates as Robbie has, so we can write:\n\n",
                "Let\u0027s use algebra to solve this problem.\n\nLet\u0027s use the following variables:\n\n* C = the number of chocolates Robbie has\n* C + 6 = the number of chocolates Danny has\n* C + 12 = the number of chocolates Arianna has\n\nWe know that Arianna has twice as many chocolates as Robbie, so we can write:\n\nC + 12 = 2 \\* C\n\nNow we can solve for C:\n\nC + 12 = 2 \\* C\n12 = C\n\nSo Robbie has 12 chocolates.\n\nDanny has 6 more"
            ]
        }
    ]
)
```

After asking the `LLM` we will obtain the `ratings`, that represent the relative rate the `LLM` assigned following the prompt, as well as the `rationale` that yielded this values:

```python
import json
print(json.dumps(next(result), indent=2))
# {
#     "generation_models": [
#         "argilla/notus-7b-v1",
#         "HuggingFaceH4/zephyr-7b-gemma-v0.1"
#     ],
#     "generations": [
#         "To find the number of chocolates Danny has, we can first write an equation that relates the number of chocolates Arianna, Danny, and Robbie have.\n\nArianna has 12 more chocolates than Danny, so we can write:\n\nArianna = Danny + 12 (1)\n\nDanny has 6 more chocolates than Robbie, so we can write:\n\nDanny = Robbie + 6 (2)\n\nArianna has twice as many chocolates as Robbie has, so we can write:\n\n",
#         "Let\u0027s use algebra to solve this problem.\n\nLet\u0027s use the following variables:\n\n* C = the number of chocolates Robbie has\n* C + 6 = the number of chocolates Danny has\n* C + 12 = the number of chocolates Arianna has\n\nWe know that Arianna has twice as many chocolates as Robbie, so we can write:\n\nC + 12 = 2 \\* C\n\nNow we can solve for C:\n\nC + 12 = 2 \\* C\n12 = C\n\nSo Robbie has 12 chocolates.\n\nDanny has 6 more"
#     ],
#     "instruction": "Arianna has 12 chocolates more than Danny. Danny has 6 chocolates more than Robbie. Arianna has twice as many chocolates as Robbie has. How many chocolates does Danny have?",
#     "ratings": [
#         2,
#         3
#     ],
#     "rationales": [
#         "The Text 1 has moderate quality. The initial mathematical relationships are correct, but the task of solving the equations and finding out the number of chocolates Danny has is left incomplete. Therefore, it doesn\u0027t fully provide the informative and accurate answer required. ",
#         "The Text 2 has good quality. The text begins to solve the problem using algebra and successfully finds the number of chocolates Robbie has. However, it does not fully answer the instruction since the number of chocolates that Danny has is not ultimately included"
#     ],
#     "ultrafeedback_model": "gpt-4"
# }
```

Let's see what this different aspects mean.

### Different aspects of UltraFeedback

The `UltraFeedback` paper proposes different types of aspect to rate the answers: `helpfulness`, `honesty`, `instruction-following`, `truthfulness`. If one want's to rate the responses according to the 4 aspects, it would imply running the `Pipeline` 4 times, incurring in more costs and time of processing. For that reason, we decided to include an extra aspect, which tries to sum up the other ones to return a special type of summary: `overall-rating`.

!!! Note
    Take a look at this task in a complete `Pipeline` at [`UltraFeedback`](../../papers/ultrafeedback.md), where you can follow the paper implementation.

## Deita Scorers

The `DEITA` paper ([`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)) includes two `Tasks` that are in charge of rating the complexity and quality of the instructions and responses generate.

!!! Note
    Take a look at this task in a complete `Pipeline` at [`DEITA`](../../papers/deita.md), where you can follow the paper implementation.

### Evol Complexity Scorer

The [`ComplexityScorer`][distilabel.steps.tasks.complexity_scorer] is in charge of assigning a score to a list of instructions based on its complexity:

```python
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks.complexity_scorer import ComplexityScorer

scorer = ComplexityScorer(
    name="complexity_scorer",
    llm=OpenAILLM(model="gpt-3.5-turbo"),
    pipeline=Pipeline(name="complexity-scorer-pipeline"),
)
scorer.load()
```

It takes a list of instructions of the following form:

```python
result = next(
    scorer.process(
        [
            {
                "instructions": [
                    "instruction 1",
                    "instruction 2"
                    "instruction 3"
                ]
            }
        ]
    )
)
```

And generates the corresponding list of scores:

```python
print(result)
# [1.0, 2.0, 1.0]
```

!!! Warning
    Keep in mind that this step can fail either due to the `LLM` not being able to return a score, or return a bad generation which isn't parseable. Using a stronger model for this task reduces the chances of this type of errors.

### Evol Quality Scorer

The second task presented in the [`DEITA`](https://arxiv.org/abs/2312.15685) paper for scoring [`QualityScorer`][distilabel.steps.tasks.quality_scorer], a pre-defined task that defines the `instruction` as the input and `score` as the output.

```python
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks.quality_scorer import QualityScorer

scorer = QualityScorer(
    name="quality_scorer",
    llm=OpenAILLM(model="gpt-3.5-turbo"),
    pipeline=Pipeline(name="quality-scorer-pipeline"),
)
scorer.load()
```

It works like the previous `ComplexityScorer` task, but works on both instruction and responses:

```python
result = next(
    scorer.process(
        [
            {
                "instructin": "instruction 1",
                "responses": [
                    "instruction 1",
                    "instruction 2"
                    "instruction 3"
                ]
            }
        ]
    )
)
```

And generates the corresponding list of scores:

```python
print(result)
# [1.0, 2.0, 1.0]
```

!!! Warning
    Keep in mind that this step can fail either due to the `LLM` not being able to return a score, or return a bad generation which isn't parseable. Using a stronger model for this task reduces the chances of this type of errors.
