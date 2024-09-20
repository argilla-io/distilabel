---
hide: toc
---
# Synthesize a social network with FinePersonas

This example will show how we can create social network interactions from FinePersonas, and fine tune different loras [MULTILORA](https://huggingface.co/blog/multi-lora-serving) to have concrete traits. [SocialAI posts from author](https://x.com/michaelsayman)

## Intro/Motivation


## Creating our SocialAI Task

Building on the new [`TextGeneration`][distilabel.steps.tasks.text_generation.TextGeneration] is easier than ever to create some custom task to generate text.

```python
from distilabel.steps.tasks import TextGeneration

class SocialAI(TextGeneration):
    follower_type: Literal["supporter", "troll", "alarmist"] = "supporter"
    system_prompt: str = (
        "You are an AI assistant expert at simulating user interactions. "
        "You must answer as if you were a '{follower_type}', be concise answer with no more than 200 characters, nothing else."
        "Here are some traits to use for your personality:\n\n"
        "{traits}"
    )
    template: str = "You are the folowing persona:\n\n{{ persona }}\n\nWhat would you say to the following?\n\n {{ post }}"
    columns: str | list[str] = ["persona", "post"]

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
        )
```


## Preparing the data

This is an example, so let's keep it short. We will use 3 posts, and 3 different types of personas. We could improve this by randomly selecting personas, or select them by their semantic similarity, but let's do it the direct way:

For each post, we will have an LLM answering it as if it was impersonating a given `persona`, 9 post-persona examples in total.

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

Let's see an example:

```python
import json
print(json.dumps(data[0], indent=4))
{
    "post": "Hmm, ok now I'm torn: should I go for healthy chicken tacos or unhealthy beef tacos for late night cravings?",
    "persona": "A high school or college environmental science teacher or an ecology student specializing in biogeography and ecosystem dynamics."
}
```

This will be our dataset, that we can ingest using the [`LoadDataFromDicts`][distilabel.steps.generators.LoadDataFromDicts]:

```python
loader = LoadDataFromDicts(data=data, batch_size=1)
```

## Simulating from different types of followers

Now that we have our data, let's see how to use our `SocialAI` task. We will use `meta-llama/Meta-Llama-3.1-70B-Instruct` as that's kind of a default lately, but it would be better to explore different types of models:

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

In the example we see that we only need to feed the follower type, and it will take care of the rest for us. We could update this too to have a random type of follower by default, and simulate from a bunch of different personalities.

## Pipeline and final dataset

TODO: Use this example, add hints

We have all the pieces to build our pipeline.

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

1. We let 


All the pieces are in place for our script, the full pipeline can be seen here for reproducibility:

??? Run

    ```python
    python examples/finepersonas_social_ai.py
    ```

```python title="finepersonas_social_ai.py"
--8<-- "examples/finepersonas_social_ai.py"
```

This is the final toy dataset we obtain: [FinePersonas-SocialAI-test](https://huggingface.co/datasets/plaguss/FinePersonas-SocialAI-test)

