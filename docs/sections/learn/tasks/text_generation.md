# Text Generation

This section will walk you through the tasks designed to generate text, from the most basic case to more involved ones like [`SelfInstruct`][distilabel.steps.tasks.self_instruct.SelfInstruct], which allows to generate instructions starting from a number of seed terms.

## The text generation task

The first `Task` that we will present is the most general one: [`TextGeneration`][distilabel.steps.tasks.text_generation.TextGeneration]. This is a pre-defined task that defines the `instruction` as the input and `generation` as the output. We can make use of this task for example with a dataset that already has all the instructions ready to be sent to be used:

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.text_generation import TextGeneration

text_generation = TextGeneration(
    name="text-generation",
    llm=MistralLLM(
        model="mistral-tiny",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    input_batch_size=8,
    pipeline=Pipeline(name="text-gen-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
text_generation.load()
```

We can just pass a simple task to test it:

```python
result = next(
    text_generation.process(
        [
            {
                "instruction": "What if the Beatles had never formed as a band?",
            },
        ] 
    )
)
```

And this is what `mistral-tiny` has to tell us to the question:

```python
print(result[0]["generation"])
# The formation of The Beatles in 1960 marked the beginning of a musical revolution that would last for decades. Their innovative songwriting, unique harmonies, and groundbreaking recordings continue to influence musicians and shape the music industry. However, had they never formed as a band, the musical landscape would have looked very different.

# In the absence of The Beatles, other bands and artists would have taken the lead in the British Invasion of the US music charts. Some possibilities include:

# 1. The Rolling Stones: The Rolling Stones emerged as a major rival to The Beatles, and they were also a crucial part of the British Invasion. They developed their own unique sound, with a raw, edgier approach that contrasted with The Beatles' more polished style.
# 2. The Who: The Who was another influential British band that rose to prominence during the same period as The Beatles. They were known for their powerful live performances and innovative approach to rock music.
# 3. Gerry & The Pacemakers: Gerry & The Pacemakers were an early Merseybeat band, and they had several hits in the UK and the US before The Beatles. Had The Beatles not formed, they might have continued to be a major force in the music world.
# 4. Cliff Richard: Cliff Richard was a successful pop singer in the UK before The Beatles, and he continued to have hits throughout the 1960s and beyond. He might have remained the dominant British pop star had The Beatles never emerged.
# 5. Motown: The Beatles' influence extended beyond rock music, and their success paved the way for other genres to gain mainstream acceptance in the US. Motown, for instance, would have faced more resistance in breaking into the US market without The Beatles' paving the way.

# It's also worth noting that The Beatles' influence extends far beyond their music. They were trendsetters in fashion, hairstyles, and cultural norms. Their break-up in 1970 marked the end of an era in popular culture, and the music industry has never been the same since. So, even if other bands and artists had taken their place, The Beatles' impact on music and culture would still be felt.
```

Let's see now how we can tweak this task to adhere a bit more to another more customized task.

### Using a custom prompt

The general `TextGeneration` task assumes our instructions need no further processing, and that we don't want to further process the response of the task for example. Let's see how we can customize the `TextGeneration` task to fit our use case.

For the following example we will implement a step presented in [`WizardLM: Empowering Large Language Models to Follow Complex Instructions`](https://arxiv.org/abs/2304.12244), which asks an `LLM` to check whether two instructions are equal or not to decide if we should keep or remove them as redundant:

```python
import string
from typing import Dict, Any, List

from distilabel.steps.tasks.text_generation import TextGeneration

system_prompt = "You are an AI judge in charge of determining the equality of two instructions. "

wizardllm_equal_prompt = """Here are two Instructions, do you think they are equal to each other and meet the following requirements?:
1. They have the same constraints and requirments.
2. They have the same depth and breadth of the inquiry.
The First Prompt: {instruction_1}
The Second Prompt: {instruction_2}
Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason):"""


class WizardLMEqualPrompts(TextGeneration):
    _template: str = wizardllm_equal_prompt

    @property
    def inputs(self) -> List[str]:
        return ["instruction_1", "instruction_2"]

    @property
    def outputs(self) -> List[str]:
        return ["response"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "system", "content": system_prompt,
                "role": "user", "content": self._template.format(**input)
            }
        ]

    def format_output(self, output: str | None, input: Dict[str, Any]) -> Dict[str, str]:
        return {"response": output.translate(str.maketrans("", "", string.punctuation))}
```

Now that we have our brand new task, let's use instantiate it (will use [`MistralLLM`][distilabel.llms.mistral.MistralLLM]):

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM

wizardlm_equality = WizardLMEqualPrompts(
    name="wizardlm_equality",
    llm=MistralLLM(
        model="mistral-small",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    input_batch_size=8,
    pipeline=Pipeline(name="wizardlm-equality-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
wizardlm_equality.load()
```

Let's ask it to compare to random instructions:

```python
result = next(
    wizardlm_equality.process(
        [
            {
                "instruction_1": "What if the Beatles had never formed as a band?",
                "instruction_2": "What are The Simpsons?"
            },
        ] 
    )
)
```

And see what we have in return:

```python
import json
print(json.dumps(result, indent=2))
# [
#   {
#     "instruction_1": "What if the Beatles had never formed as a band?",
#     "instruction_2": "What are The Simpsons?",
#     "model_name": "mistral-small",
#     "response": "Not Equal The first prompt is a counterfactual question that invites exploration of the Beatles impact on music and culture if they had not formed while the second prompt asks for an explanation or definition of a longrunning TV show The Simpsons They do not share the same constraints requirements depth or breadth of inquiry"
#   }
# ]
```

!!! Note
    We can see the `response` contais "Not Equal", but it didn't strictly followed the prompt as requested. This can be a hint that a more powerful model is required, or the prompt needs some extra tuning.

## Guided text generation

Other than the base generation tasks, the literature has proposed some `Tasks` to provide extra functionality, like the following:

### Self Instruct

The first we are going to look at is [`SelfInstruct`][distilabel.steps.tasks.self_instruct.SelfInstruct]. This pre-defined task is inspired by [`Self-Instruct: Aligning Language Models with Self-Generated Instructions`](https://arxiv.org/abs/2212.10560), and has the following intent: given a number of instructions, a certain criteria for query generations, an application description, and an input, generates a number of instruction related to the given input and following what is stated in the criteria for query generation and the application description.

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.self_instruct import SelfInstruct

self_instruct = SelfInstruct(
    name="text-generation",
    num_instructions=3,
    input_batch_size=8,
    llm=MistralLLM(
        model="mistral-medium",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    pipeline=Pipeline(name="self-instruct-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
self_instruct.load()
```

Let's pass a simple input:

```python
result = next(
    self_instruct.process(
        [
            {
                "input": "What are fantasy novels?",
            },
        ] 
    )
)
```

And this is what we have in return:

```python
import json
print(json.dumps(result[0]["instructions"], indent=2))
# [
#   "1. Can you explain the common elements found in fantasy novels and their role in storytelling?",
#   "2. Generate a brief summary of a popular fantasy novel, highlighting its unique features.",
#   "3. Compare and contrast the world-building techniques used by two renowned fantasy authors."
# ]
```

By tweaking the `num_instructions` and the `criteria_for_query_generation` we can see how this is a really powerful `Task` to generate synthetic data starting from a small amount of initial instructions.

### Evol Instruct

[`EvolInstruct`][distilabel.steps.tasks.evol_instruct.base.EvolInstruct]

#### Evol Complexity

[`EvolComplexity`][distilabel.steps.tasks.evol_instruct.evol_complexity.base.EvolComplexity]

### Evol Quality

[`EvolQuality`][distilabel.steps.tasks.evol_quality.base.EvolQuality]

```python
llm=MistralLLM(
    model="mistral-tiny",
    api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
)
```
