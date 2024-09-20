---
hide: toc
---

# Create a social network with FinePersonas

In this example, we'll explore the creation of specialized user personas for social network interactions using the [FinePersonas-v0.1](https://huggingface.co/datasets/argilla/FinePersonas-v0.1) dataset from Hugging Face. The final dataset will be ready to fine-tune a chat model with specific traits and characteristics.

## Introduction

We'll delve into the process of fine-tuning different LoRA (Low-Rank Adaptation) models to imbue these personas with specific traits and characteristics.

This approach draws inspiration from Michael Sayman's work on [SocialAI](https://apps.apple.com/us/app/socialai-ai-social-network/id6670229993) (visit the [profile](https://x.com/michaelsayman) to see some examples), to leverage [FinePersonas-v0.1](https://huggingface.co/datasets/argilla/FinePersonas-v0.1) for building models that can emulate bots with specific behaviour.

By fine-tuning these adapters, we can potentially create AI personas with distinct characteristics, communication styles, and areas of expertise. The result? AI interactions that feel more natural and tailored to specific contexts or user needs. For those interested in the technical aspects of this approach, we recommend the insightful blog post on [Multi-LoRA serving](https://huggingface.co/blog/multi-lora-serving). It provides a clear and comprehensive explanation of the technology behind this innovative method.

Let's jump to the demo.

## Creating our SocialAI Task

Building on the new [`TextGeneration`](https://distilabel.argilla.io/dev/components-gallery/tasks/textgeneration/), creating custom tasks is easier than ever before. This powerful tool opens up a world of possibilities for creating tailored text-based content with ease and precision. We will create a `SocialAI` task that will be in charge of generating responses to user interactions, taking into account a given `follower_type`, and use the perspective from a given `persona`:

```python
from distilabel.steps.tasks import TextGeneration

class SocialAI(TextGeneration):
    follower_type: Literal["supporter", "troll", "alarmist"] = "supporter"
    system_prompt: str = (
        "You are an AI assistant expert at simulating user interactions. "
        "You must answer as if you were a '{follower_type}', be concise answer with no more than 200 characters, nothing else."
        "Here are some traits to use for your personality:\n\n"
        "{traits}"
    )  # (1)
    template: str = "You are the folowing persona:\n\n{{ persona }}\n\nWhat would you say to the following?\n\n {{ post }}"  # (2)
    columns: str | list[str] = ["persona", "post"]  # (3)

    _follower_traits: dict[str, str] = {
        "supporter": (
            "- Encouraging and positive\n"
            "- Tends to prioritize enjoyment and relaxation\n"
            "- Focuses on the present moment and short-term pleasure\n"
            "- Often uses humor and playful language\n"
            "- Wants to help others feel good and have fun\n"
        ),
        "troll": (
            "- Provocative and confrontational\n"
            "- Enjoys stirring up controversy and conflict\n"
            "- Often uses sarcasm, irony, and mocking language\n"
            "- Tends to belittle or dismiss others' opinions and feelings\n"
            "- Seeks to get a rise out of others and create drama\n"
        ),
        "alarmist": (
            "- Anxious and warning-oriented\n"
            "- Focuses on potential risks and negative consequences\n"
            "- Often uses dramatic or sensational language\n"
            "- Tends to be serious and stern in tone\n"
            "- Seeks to alert others to potential dangers and protect them from harm (even if it's excessive or unwarranted)\n"
        ),
    }

    def load(self) -> None:
        super().load()
        self.system_prompt = self.system_prompt.format(
            follower_type=self.follower_type,
            traits=self._follower_traits[self.follower_type]
        )  # (4)
```

1. We have a custom system prompt that will depend on the `follower_type` we decide for our model.

2. The base template or prompt will answert to the `post` we have, from the point of view of a `persona`.

3. We will need our dataset to have both `persona` and `post` columns to populate the prompt.

4. In the load method we place the specific traits for our follower type in the system prompt.

## Data preparation

This is an example, so let's keep it short. We will use 3 posts, and 3 different types of personas. While there's potential to enhance this process (perhaps by implementing random persona selection or leveraging semantic similarity) we'll opt for a straightforward method in this demonstration.

Our goal is to create a set of nine examples, each pairing a post with a persona. To achieve this, we'll employ an LLM to respond to each post from the perspective of a specific `persona`, effectively simulating how different characters might engage with the content.

```python
posts = [
    {
        "post": "Hmm, ok now I'm torn: should I go for healthy chicken tacos or unhealthy beef tacos for late night cravings?"
    },
    {
        "post": "I need to develop a training course for my company on communication skills. Need to decide how deliver it remotely."
    },
    {
        "post": "I'm always 10 minutes late to meetups but no one's complained. Could this be annoying to them?"
    },
]

personas = (
    load_dataset("argilla/FinePersonas-v0.1-clustering-100k", split="train")
    .shuffle()
    .select(range(3))
    .select_columns("persona")
    .to_list()
)

data = []
for post in posts:
    for persona in personas:
        data.append({"post": post["post"], "persona": persona["persona"]})
```

Each row in will have the following format:

```python
import json
print(json.dumps(data[0], indent=4))
{
    "post": "Hmm, ok now I'm torn: should I go for healthy chicken tacos or unhealthy beef tacos for late night cravings?",
    "persona": "A high school or college environmental science teacher or an ecology student specializing in biogeography and ecosystem dynamics."
}
```

This will be our dataset, that we can ingest using the [`LoadDataFromDicts`](https://distilabel.argilla.io/dev/components-gallery/steps/loaddatafromdicts/):

```python
loader = LoadDataFromDicts(data=data)
```

## Simulating from different types of followers

With our data in hand, we're ready to explore the capabilities of our SocialAI task. For this demonstration, we'll make use of of `meta-llama/Meta-Llama-3.1-70B-Instruct`
While this model has become something of a go-to choice recently, it's worth noting that experimenting with a variety of models could yield even more interesting results:

```python
from distilabel.llms import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 256,
    },
)
follower_type = "supporter"

follower = SocialAI(
    llm=llm,
    follower_type=follower_type,
    name=f"{follower_type}_user",
)
```

This setup simplifies the process, we only need to input the follower type, and the system handles the rest. We could update this too to have a random type of follower by default, and simulate from a bunch of different personalities.

## Building our Pipeline

The foundation of our pipeline is now in place. At its core is a single, powerful LLM. This versatile model will be repurposed to drive three distinct `SocialAI` Tasks, each tailored to a specific `TextGeneration` task, and each one of them will be prepared for Supervised Fine Tuning using [`FormatTextGenerationSFT`](https://distilabel.argilla.io/dev/components-gallery/steps/formattextgenerationsft/):

```python
with Pipeline(name="Social AI Personas") as pipeline:
    loader = LoadDataFromDicts(data=data, batch_size=1)

    llm = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        generation_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 256,
        },
    )

    for follower_type in ["supporter", "troll", "alarmist"]:
        follower = SocialAI(
            llm=llm,
            follower_type=follower_type,
            name=f"{follower_type}_user",  # (1)
            output_mappings={
                "generation": f"interaction_{follower_type}"  # (2)
            }
        )
        format_sft = FormatTextGenerationSFT(
            name=f"format_sft_{follower_type}",
            input_mappings={
                "instruction": "post",
                "generation": f"interaction_{follower_type}"  # (3)
            },
        )
        loader >> follower >> format_sft  # (4)
```

1. We update the name of the step to keep track in the pipeline.

2. The `generation` column from each LLM will be mapped to avoid them being overriden, as we are reusing the same task.

3. As we have modified the output column from `SocialAI`, we redirect each one of the "follower_type" responses.

4. Connect the loader to each one of the follower tasks and `format_sft` to obtain 3 different subsets.

The outcome of this pipeline will be three specialized models, each fine-tuned to a unique `follower type` crafted by the `SocialAI` task. These models will generate SFT-formatted datasets, where each post is paired with its corresponding interaction data for a specific follower type. This setup enables seamless fine-tuning using your preferred framework, such as [TRL](https://huggingface.co/docs/trl/index), or any other training framework of your choice.

## Script and final dataset

All the pieces are in place for our script, the full pipeline can be seen here:

??? Run

    ```python
    python examples/finepersonas_social_ai.py
    ```

```python title="finepersonas_social_ai.py"
--8<-- "examples/finepersonas_social_ai.py"
```

This is the final toy dataset we obtain: [FinePersonas-SocialAI-test](https://huggingface.co/datasets/plaguss/FinePersonas-SocialAI-test)

You can see examples of how to load each subset of them to fine-tune a model:

```python
from datasets import load_dataset

ds = load_dataset("plaguss/FinePersonas-SocialAI-test", "format_sft_troll")
```

And a sample of the generated field with the corresponding `post` and `persona`:

```json
{
    "post": "Hmm, ok now I\u0027m torn: should I go for healthy chicken tacos or unhealthy beef tacos for late night cravings?",
    "persona": "A high school or undergraduate physics or chemistry teacher, likely with a focus on experimental instruction.",
    "interaction_troll": "\"Late night cravings? More like late night brain drain. Either way, it\u0027s just a collision of molecules in your stomach. Choose the one with more calories, at least that\u0027s some decent kinetic energy.\"",
}
```

There's a lot of room for improvement, but quite a promising start.
