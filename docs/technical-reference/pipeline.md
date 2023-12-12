# Pipelines

This section will detail the [`Pipeline`][distilabel.pipeline.Pipeline], providing guidance on creating and using them.

## Pipeline

The [Pipeline][distilabel.pipeline.Pipeline] class is a central component in `distilabel`, responsible for crafting datasets. It manages the generation of datasets and oversees the interaction between the generator and labeller `LLMs`.

You create an instance of the [`Pipeline`][distilabel.pipeline.Pipeline] by providing a *generator* and an optional *labeller* [LLM][distilabel.llm.base.LLM]. Interactions with it are facilitated through its `generate` method. This method requires a [`dataset`](https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset), specifies the *num_generations* to determine the number of examples to be created, and includes additional parameters for controlling the *batch_size* and managing the generation process.

Let's start by a Pipeline with a single `LLM` as a generator.

### Generator

We will create a [`Pipeline`][distilabel.pipeline.Pipeline] that will use [Notus](https://huggingface.co/argilla/notus-7b-v1) from a Huggingface [Inference Endpoint][distilabel.llm.InferenceEndpointsLLM]. For this matter, we need to create a [TextGenerationTask][distilabel.tasks.TextGenerationTask], and specify the format we want to use for our `Prompt`, in this case *notus*, which corresponds to the same for *zephyr*.

```python
import os

from distilabel.tasks.prompt import Prompt

class NotusTextGenerationTask(TextGenerationTask):
    def generate_prompt(self, input: str) -> str:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input,
        ).format_as("notus")

pipe_generation = Pipeline(
    generator=InferenceEndpointsLLM(
        endpoint_name=endpoint_name,  # The name given of the deployed model
        endpoint_namespace=endpoint_namespace,  # This usually corresponds to the organization, in this case "argilla"
        token=token,  # hf_...
        task=NotusTextGenerationTask(),
        max_new_tokens=512,
        do_sample=True
    )
)
```

We've set up our pipeline using a specialized [`TextGenerationTask`](distilabel.tasks.text_generation.base.TextGenerationTask) (refer to the [tasks section](./tasks.md) for more task details), and an [InferenceEndpointsLLM][distilabel.llm.huggingface.inference_endpoints.InferenceEndpointsLLM] configured for [`notus-7b-v1`](https://huggingface.co/argilla/notus-7b-v1), although any of the available `LLMs` will work.

To utilize the [Pipeline][distilabel.pipeline.Pipeline] for dataset generation, we call the generate method. We provide it with the input dataset and specify the desired number of generations. In this example, we've prepared a `Dataset` with a single row to illustrate the process. This dataset contains one row, and we'll trigger 2 generations from it:

```python
from datasets import Dataset

dataset = Dataset.from_dict({"input": ["Create an easy dinner recipe with few ingredients"]})
dataset_generated = pipe_generation.generate(dataset, num_generations=2)
```

Now, let's examine the dataset that was generated. It's a [`CustomDataset`][distilabel.dataset.CustomDataset], equipped with additional features for seamless interaction with [`Argilla`](https://github.com/argilla-io/argilla).

```python
print(dataset_generated)
# Dataset({
#     features: ['input', 'generation_model', 'generation_prompt', 'raw_generation_responses', 'generations'],
#     num_rows: 1
# })

print(dataset_generated[0]["generations"][0])
# Here's a simple and delicious dinner recipe with only a few ingredients:

# Garlic Butter Chicken with Roasted Vegetables

# Ingredients:
# - 4 boneless, skinless chicken breasts
# - 4 tablespoons butter
# - 4 cloves garlic, minced
# - 1 teaspoon dried oregano
# - 1/2 teaspoon salt
# - 1/4 teaspoon black pepper
# - 1 zucchini, sliced
# - 1 red bell pepper, sliced
# - 1 cup cherry tomatoes

# Instructions:

# 1. Preheat oven to 400°F (200°C).

# 2. Melt butter in a small saucepan over low heat. Add minced garlic and heat until fragrant, about 1-2 minutes.

# 3. Place chicken breasts in a baking dish and brush garlic butter over each one.

# 4. Sprinkle oregano, salt, and black pepper over the chicken.

# 5. In a separate baking dish, add sliced zucchini, red bell pepper, and cherry tomatoes. Brush with remaining garlic butter.

# 6. Roast the chicken and vegetables in the preheated oven for 25-30 minutes or until cooked through and the vegetables are tender and lightly browned.

# 7. Transfer the chicken to plates and serve with the roasted vegetables alongside. Enjoy!

# This recipe requires simple ingredients and is easy to prepare, making it perfect for a quick, satisfying dinner. The garlic butter adds maximum flavor, while the roasted vegetables complement the chicken beautifully, providing additional nutrition and texture. With minimal effort, you can have a delicious and balanced meal on the table in no time.
```

### Labeller

Next, we move on to labelLing a dataset. Just as before, we need an `LLM` for our `Pipeline`. In this case we will use [`OpenAILLM`][distilabel.llm.openai.OpenAILLM] with `gpt-4`, and a `PreferenceTask`, [UltraFeedbackTask][distilabel.tasks.preference.ultrafeedback.UltraFeedbackTask] for instruction following.

```python
from distilabel.pipeline import Pipeline
from distilabel.llm import OpenAILLM
from distilabel.tasks import UltraFeedbackTask

pipe_labeller = Pipeline(
    labeller=OpenAILLM(
        model="gpt-4",
        task=UltraFeedbackTask.for_instruction_following(),
        max_new_tokens=256,
        num_threads=8,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    ),
)
```

For this example dataset, we've extracted 2 sample rows from the [UltraFeedback binarized dataset](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences), formatted as expected by the default `LLM` and `Task`.

We've selected two distinct examples, one correctly labeled and the other incorrectly labeled in the original dataset. In this instance, the `dataset` being generated includes two columns: the *input*, as seen in the generator, and a *generations* column containing the model's responses.

```python
from datasets import Dataset

dataset_test = Dataset.from_dict(
    {
        "input": [
            "Describe the capital of Spain in 25 words.",
            "Design a conversation between a customer and a customer service agent."
        ],
        "generations": [
            ["Santo Domingo is the capital of Dominican Republic"],
            ["Customer: Hello, I'm having trouble with my purchase.\n\nCustomer Service Agent: I'm sorry to hear that. Could you please tell me more about the issue you are facing?\n\nCustomer: Yes, I ordered a pair of shoes from your company a week ago, but I haven't received them yet.\n\nCustomer Service Agent: I apologize for the inconvenience. Could you please provide me with your order number and full name so I can look into this for you?\n\nCustomer: Sure, my name is John Doe and my order number is ABCD1234.\n\nCustomer Service Agent: Thank you, John. I have checked on your order and it appears that it is still being processed. It should be shipped out within the next 24 hours.\n\nCustomer: That's good to hear, but can you also tell me the expected delivery time?\n\nCustomer Service Agent: Absolutely, based on your location, the estimated delivery time is 3-5 business days after shipping. You will receive a tracking number via email once the item is shipped, which will provide real-time updates on your package.\n\nCustomer: Thanks for the information. One more thing, what is your return policy if the shoes don't fit?\n\nCustomer Service Agent: Our company offers a 30-day return policy. If you are not satisfied with the product or if it doesn't fit, you can return it for a full refund or an exchange within 30 days of delivery. Please keep in mind that the product must be in its original packaging and in the same condition as when you received it.\n\nCustomer: Okay, that's good to know. Thank you for your help.\n\nCustomer Service Agent: You're welcome, John. I'm glad I could assist you. If you have any further questions or concerns, please don't hesitate to reach out to us. Have a great day!"]
        ] 
    }
)

ds_labelled = pipe_labeller.generate(dataset_test)
```

Let's select the relevant columns from the labelled dataset, and take a look at the first record. This allows us to observe the *rating* and the accompanying *rationale* that provides an explanation.

```python
ds_labelled.select_columns(["input", "generations", "rating", "rationale"])[0]
{'input': 'Describe the capital of Spain in 25 words.',
 'generations': ['Santo Domingo is the capital of Dominican Republic'],
 'rating': [1.0],
 'rationale': ['The text is irrelevant to the instruction. It describes the capital of the Dominican Republic instead of Spain.']}
```

### Generator and Labeller

In the final scenario, we have a [`Pipeline`][distilabel.pipeline.Pipeline] utilizing both a *generator* and a *labeller* `LLM`. Once more, we'll employ the [Inference Endpoint][distilabel.llm.InferenceEndpointsLLM] with `notus-7b-v1` for the *generator*, using a different *system prompt* this time. As for the labeller, we'll use `gpt-3.5-turbo`, which will label the examples for *instruction following*.

```python
pipe_full = Pipeline(
    generator=InferenceEndpointsLLM(
        endpoint_name=endpoint_name,
        endpoint_namespace=endpoint_namespace,
        token=token,
        task=NotusTextGenerationTask(
            system_prompt="You are an expert writer of XKCD, a webcomic of romance, sarcasm, math, and language."
        ),
        max_new_tokens=512,
        do_sample=True
    ),
    labeller=OpenAILLM(
        model="gpt-3.5-turbo",
        task=UltraFeedbackTask.for_instruction_following(),
        max_new_tokens=256,
        num_threads=4,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    ),
)
```

For this example, we'll set up a pipeline to generate and label a dataset of short stories inspired by [XKCD](https://xkcd.com/). To do this, we'll define the *system_prompt* for the `NotusTextGenerationTask`. The dataset will follow the same format we used for the generator scenario, featuring an *input* column with the examples, in this case, just one.

```python
xkcd_instructions = Dataset.from_dict(
    {
        "input": [            
            "Could you imagine an interview process going sideways?"
        ]
    }
)
ds_xkcd = pipe_full.generate(xkcd_instructions, num_generations=3)
```

We will now take a look to one of the *generations*, along with the *rating* and *rational* given by our *labeller* `LLM`:

```python
print(ds_xkcd[1]["generations"][0])
print("-----" * 5)
print("RATING: ", ds_xkcd[1]["rating"][0])
print("RATIONALE: ", ds_xkcd[1]["rationale"][0])

# Yes, absolutely! Here's a fictional interview scenario turned into an XKCD-style comic:

# (Interviewee meets with an unsmiling interviewer)

# Interviewer: Good morning! Have a seat. Tell me about your experience working with teams.

# Interviewee: Well, I've worked in large teams on group projects before. It could be challenging, but we always managed to pull through.

# (Smugly) Interviewer: Challenging, huh? (tapping pen on desk) And how did you manage to overcome these challenges?

# Interviewee: (confidently) Communication was key. I made sure to stay in touch with the team and keep everyone updated on our progress.

# Interviewer: Communication. Hm. And what if communication failed?

# Interviewee: (thrown off balance) Well, I mean...there was one time when we couldn't connect via video call. But we picked up the phone, and we all understood what needed to be done.

# Interviewer: But what if the communication on the technical level failed, say, like a computer system with a software glitch?

# Interviewee: (feeling the pressure) That's never happened to me before, but if it did, we would have to troubleshoot and find a workaround, right?

# Interviewer: (smirking) Oh, but finding a workaround could mean delegating responsibilities among the team, which requires communication. It's a vicious cycle!

# (Interviewee visibly uncomfortable)

# Interviewer: And what if there was a communication breakdown among the team members themselves?

# Interviewee: (unsure) I think we would try to sort it out as soon as possible to avoid any further problems.

# Interviewer: (sarcastically) Yes, avoiding further problems is critical. Don't want to let those deadlines slip, do we?

# (Interviewer types frantically on their computer keyboard)

# Interviewer: (softly but wordily) Note to self: Avoid this candidate for team projects.

# (The interviewer returns his attention back to the interviewee)

# Interviewer: Well, moving on...
# -------------------------
# RATING:  4.0
# RATIONALE:  The text provides a fictional interview scenario that aligns with the task goal of imagining an interview process going sideways. It includes dialogue between an interviewer and interviewee, showcasing a breakdown in communication and the interviewer's sarcastic and dismissive attitude towards the interviewee's responses.
```

## pipeline

Considering recurring patterns in dataset creation, we can facilitate the process by utilizing the [`Pipeline`][distilabel.pipeline.Pipeline]. This is made simpler through the [`pipeline`][distilabel.pipeline.pipeline] function, which provides the necessary parameters for creating a `Pipeline`.

In the code snippet below, we use the [`pipeline`][distilabel.pipeline.pipeline] function to craft a `pipeline` tailored for a *preference task*, specifically focusing on *text-quality* as the *subtask*. If we don't initially provide a *labeller* [`LLM`][distilabel.llm.base.LLM], we can specify the subtask we want our `pipeline` to address. By default, this corresponds to [`UltraFeedbackTask`][distilabel.tasks.preference.ultrafeedback.UltraFeedbackTask]. It's mandatory to specify the generator of our choice; however, the labeller defaults to `gpt-3.5-turbo`. Optional parameters required for [OpenAILLM][distilabel.llm.openai.OpenAILLM] can also be passed as optional keyword arguments.

```python
from distilabel.pipeline import pipeline

pipe = pipeline(
    "preference",
    "text-quality",
    generator=InferenceEndpointsLLM(
        endpoint_name=endpoint_name,
        endpoint_namespace=endpoint_namespace,
        token=token,
        task=NotusTextGenerationTask(),
        max_new_tokens=512,
        do_sample=True
    ),
    max_new_tokens=256,
    num_threads=2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)
```


For the dataset, we'll begin with three rows from [HuggingFaceH4/instruction-dataset](https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset). We'll request two generations with checkpoints enabled to safeguard the data in the event of any failures, which is the default behavior.

```python
instruction_dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:3]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

pipe_dataset = pipe.generate(
    instruction_dataset,
    num_generations=2,
    batch_size=1,
    enable_checkpoints=True,
    display_progress_bar=True,
)
```

Finally, let's see one of the examples from the dataset:

```python
print(pipe_dataset["input"][-1])
# Create a 3 turn conversation between a customer and a grocery store clerk - that is, 3 per person. Then tell me what they talked about.

print(pipe_dataset["generations"][-1][-1])
# Customer: Hi there, I'm looking for some fresh berries. Do you have any raspberries or blueberries in stock?

# Grocery Store Clerk: Yes, we have both raspberries and blueberries in stock today. Would you like me to grab some for you or can you find them yourself?

# Customer: I'd like your help getting some berries. Can you also suggest which variety is sweeter? Raspberries or blueberries?

# Grocery Store Clerk: Raspberries and blueberries both have distinct flavors. Raspberries are more tart and a little sweeter whereas blueberries tend to be a little sweeter and have a milder taste. It ultimately depends on your personal preference. Let me grab some of each for you to try at home and see which one you like better.

# Customer: That sounds like a great plan. How often do you receive deliveries? Do you have some new varieties of berries arriving soon?

# Grocery Store Clerk: We receive deliveries twice a week, on Wednesdays and Sundays. We also have a rotation of different varieties of berries throughout the season, so keep an eye out for new arrivals. Thanks for shopping with us, can I help you with anything else today?

# Customer: No, that's all for now. I'm always happy to support your local store.

# turn 1: berries, fresh produce availability, customer preference
# turn 2: product recommendations based on taste and personal preference, availability
# turn 3: store acknowledgment, shopping gratitude, loyalty and repeat business expectation.

print(pipe_dataset["rating"][-1][-1])
# 5.0

print(pipe_dataset["rationale"][-1][-1])
# The text accurately follows the given instructions and provides a conversation between a customer and a grocery store clerk. The information provided is correct, informative, and aligned with the user's intent. There are no hallucinations or misleading details.
```

The API reference can be found here: [pipeline][distilabel.pipeline.pipeline]

## Argilla integration

The [CustomDataset][distilabel.dataset.CustomDataset] generated entirely by AI models may require some additional human processing. To facilitate human feedback, the dataset can be uploaded to [`Argilla`](https://github.com/argilla-io/argilla). This process involves logging into an [`Argilla`](https://docs.argilla.io/en/latest/getting_started/cheatsheet.html#connect-to-argilla) instance, converting the dataset to the required format using `CustomDataset.to_argilla()`, and subsequently using push_to_argilla on the resulting dataset:

```python
import argilla as rg

rg.init(
    api_key="<YOUR_ARGILLA_API_KEY>",
    api_url="<YOUR_ARGILLA_API_URL>"
)

rg_dataset = pipe_dataset.to_argilla()  # Any dataset resulting from `Pipeline.generate`
rg_dataset.push_to_argilla(name="preference-dataset", workspace="admin")
```
