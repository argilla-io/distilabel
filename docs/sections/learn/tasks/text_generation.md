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
    pipeline=Pipeline(name="text-gen-pipeline"),
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

Additionally, we can also define the [`ChatGeneration`][distilabel.steps.tasks.text_generation.ChatGeneration] task, which has the same goal as the `TextGeneration` task, but expects a list of messages from a conversation formatted using the [OpenAI format](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models) instead of a single instruction, so that the last assistant response is generated no matter how long the conversation is.

So that the task can be used in the same way as the previous one, we can instantiate it as follows:

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.text_generation import ChatGeneration

chat_generation = ChatGeneration(
    name="chat-generation",
    llm=MistralLLM(
        model="mistral-tiny",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    input_batch_size=8,
    pipeline=Pipeline(name="text-gen-pipeline"),
)

# remember to call .load() if testing outside of a Pipeline context
chat_generation.load()
```

We can just pass a simple task to test it:

```python
result = next(
    text_generation.process(
        [
            {
                "messages": [
                    {"role": "user", "content": "What if the Beatles had never formed as a band?"},
                ],
            },
        ] 
    )
)

```

Additionally, note that both `TextGeneration` and `ChatGeneration` tasks admit an optional `system_prompt` parameter that can be used to prepend a system message to the input instruction or conversation. This can be useful to provide context or additional information to the model, as shown in the following example:

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.text_generation import ChatGeneration

chat_generation = ChatGeneration(
    name="chat-generation",
    llm=MistralLLM(
        model="mistral-tiny",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    system_prompt="You are a helpful assistant, that knows everything about The Beatles.",
    input_batch_size=8,
    pipeline=Pipeline(name="text-gen-pipeline"),
)

# remember to call .load() if testing outside of a Pipeline context
chat_generation.load()
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
1. They have the same constraints and requirements.
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
    We can see the `response` contains "Not Equal", but it didn't strictly followed the prompt as requested. This can be a hint that a more powerful model is required, or the prompt needs some extra tuning.

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

This `Task` was defined in [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244), the idea is, starting from a series of initial instructions, evolve them according to a scheme of prompts to obtain more complex ones. Let's see how we can use [`EvolInstruct`][distilabel.steps.tasks.evol_instruct.base.EvolInstruct]:

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.evol_instruct.base import EvolInstruct

evol_instruct = EvolInstruct(
    name="evol-instruct",
    num_evolutions=2,
    store_evolutions=True,
    input_batch_size=8,
    llm=MistralLLM(
        model="mistral-small",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    pipeline=Pipeline(name="evol-instruct-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
evol_instruct.load()
```

We can now use a sample instruction to see what that yields:

```python
result = next(
    evol_instruct.process(
        [
            {
                "instruction": "What are fantasy novels?",
            },
        ] 
    )
)
```

!!! Note
    As we used `store_evolutions=True`, we now can see the evolution from the starting point. Remember to visit the [`API Reference`][distilabel.steps.tasks.evol_instruct.base.EvolInstruct] to take into account all the parameters.

Let's see the evolved instructions we obtained:

```python
import json
print(json.dumps(result[0]["evolved_instructions"], indent=2))
# [
#   "Can you name some lesser-known literary genres that explore imaginary worlds, magical elements, and epic adventures, similar to fantasy novels, but with a unique twist and a smaller readership?",
#   "How about you suggest some under-the-radar literary categories that, much like fantasy literature, delve into fictional realms, incorporate magical aspects, and narrate grand journeys, but with a distinct flavor and a more limited, devoted readership?"
# ]
```

This strategy of evolving a set of instructions synthetically has yielded strong results as can be seen in the original paper, leading to the family of [WizardLM](https://github.com/nlpxucan/WizardLM).

We will now take a look at some `EvolInstruct` *inspired* tasks that have been used for [`DEITA`]([`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)) datasets.

#### Evol Complexity

[`EvolComplexity`][distilabel.steps.tasks.evol_instruct.evol_complexity.base.EvolComplexity] evolves the instructions to make them specifically more complex. It follows the evolutionary approach from `EvolInstruct` but with slightly different prompts.

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.evol_instruct.evol_complexity.base import EvolComplexity

evol_complexity = EvolComplexity(
    name="evol-complexity",
    num_evolutions=1,
    input_batch_size=8,
    llm=MistralLLM(
        model="mistral-small",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    pipeline=Pipeline(name="evol-complexity-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
evol_complexity.load()
```

Let's see it with the same previous example from `EvolInstruct`, this time with a single evolution and keeping the last generation:

```python
result = next(
    evol_complexity.process(
        [
            {
                "instruction": "What are fantasy novels?",
            },
        ] 
    )
)
```

This would be the evolved instruction:

```python
print(result[0]["evolved_instruction"])
# Could you explain the literary genre of fantasy novels, providing examples and discussing how they differ from science fiction?

# (Note: I added a requirement to discuss the differences between fantasy novels and science fiction, and tried to keep the prompt reasonably concise.)
```

And we have similar results to what we obtained with `EvolInstruct`, with a slight modification in the inner prompts used.

!!! Note
    Take into account there isn't just randomness from the `LLM`, but also from the mutation selected (the prompt used to evolve the instruction).

#### Evol Quality

The [`EvolQuality`][distilabel.steps.tasks.evol_quality.base.EvolQuality] `Task` appeared in [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685), as a posterior step to the previous [`EvolComplexityGenerator`][distilabel.steps.tasks.evol_instruct.evol_complexity.generator.EvolComplexityGenerator]. It takes a different approach: we evolve the **quality** of the **responses** given a prompt. Let's see an example:

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.evol_quality.base import EvolQuality

evol_quality = EvolQuality(
    name="evol-quality",
    num_evolutions=1,
    input_batch_size=8,
    llm=MistralLLM(
        model="mistral-small",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    pipeline=Pipeline(name="evol-quality-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
evol_quality.load()
```

We will use it on the output from `EvolComplexity` task:

```python
result = next(
    evol_quality.process(
        [
            {
                "instruction": "What are fantasy novels?",
                "response": "Could you explain the literary genre of fantasy novels, providing examples and discussing how they differ from science fiction?\n\n(Note: I added a requirement to discuss the differences between fantasy novels and science fiction, and tried to keep the prompt reasonably concise.)"
            },
        ] 
    )
)
```

And we obtain in return an evolution from the previous response with a *mutation* applied to the response:

```python
print(result[0]["evolved_response"])
# Fantasy novels are a captivating genre of literature, immersing readers in imaginary worlds filled with magical elements, mythical creatures, and epic adventures. They often feature complex plotlines and unique characters, offering a delightful escape from reality.
```

!!! Note
    Take into account that just as we had with the `EvolComplexity` task, there is randomness involved with the inner mutation prompt used.
