# DEITA

[DEITA (Data-Efficient Instruction Tuning for Alignment)](https://arxiv.org/abs/2312.15685) studies an automatic data selection process by first quantifying the data quality based on complexity, quality and diversity. Second, select the best potential combination from an open-source dataset that would fit into the budget you allocate to tune your own LLM.

In most setting we cannot allocate unlimited resources for instruction-tuning LLMs. Therefore, the DEITA authors investigated how to select qualitative data for instruction tuning based on the principle of fewer high-quality samples. Liu et al. tackle the issue of first defining good data and second identifying it to respect an initial budget to instruct-tune your LLM.

The strategy utilizes **LLMs to replace human effort in time-intensive data quality **tasks on **instruction-tuning** datasets**. DEITA introduces a way to measure data quality across three critical dimensions: complexity, quality and diversity.

![DEITA pipeline overview](../../../assets/tutorials-assets/deita/overview.png)

You can see that we see again the dataset of instructions/responses and we kind of reproducing the second step when we learn how to optimize the responses according to an instruction by comparing several possibilities.

![DEITA pipeline overview](../../../assets/pipelines/deita.png)

### Datasets and budget

We will dive deeper into the whole process. We will investigate each stage to efficiently select the final dataset used for supervised fine-tuning with a budget constraint. We will tackle technical challenges by explaining exactly how you would assess good data as presented in the paper.

As a reminder, we're looking for a strategy to automatically select good data for the instruction-tuning step when you want to fine-tune an LLM to your own use case taking into account a resource constraint. This means that you cannot blindly train a model on any data you encounter on the internet.

The DEITA authors assume that you have access to open-source datasets that fit your use case. This may not be the case entirely. But with open-source communities tackling many use cases, with projects such as [BLOOM](https://arxiv.org/pdf/2110.08207.pdf) or [AYA](https://cohere.com/research/aya), it's likely that your use case will be tackled at some point. Furthermore, you could generate your own instruction/response pairs with methods such as [self-generated instructions](https://aclanthology.org/2023.acl-long.754/) using distilabel. This tutorial assumes that we have a data pool with excessive samples for the project's cost constraint. In short, we aim to achieve adequate performance from fewer samples.

The authors claim that the subsample size "correlates proportionally with the computation consumed in instruction tuning". Hence on a first approximation, reducing the sample size means reducing computation consumption and so the total development cost. Reproducing the paper notations, we will associate the budget m to a number of instruction/response pairs that you can set depending on your real budget.

![Datasets table](../../../assets/tutorials-assets/deita/datasets.png)

To match the experimental set-up, dataset *X\_sota* is a meta-dataset combining major open-source datasets available to instruct-tune LLMs. This dataset is composed of [ShareGPT](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) (58k instruction/response pairs), [UltraChat](https://huggingface.co/datasets/stingning/ultrachat) (105k instruction/response pairs) and [WizardLM](https://github.com/nlpxucan/WizardLM) (143k instruction/response pairs). It sums to more than 300k instruction/response pairs. We aim to reduce the final subsample to 6k instruction/response pairs.

## Setup the notebook and packages

Let's prepare our dependencies:

```bash
pip install "distilabel[openai,hf-transformers]>=1.0.0"
pip install pynvml huggingface_hub argilla
```

Import distilabel:

```python
from distilabel.models import TransformersLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import ConversationTemplate, DeitaFiltering, ExpandColumns, LoadDataFromHub
from distilabel.steps.tasks import ComplexityScorer, EvolInstruct, EvolQuality, GenerateEmbeddings, QualityScorer
```

Define the distilabel Pipeline and load the dataset from the Hugging Face Hub.

```python
pipeline = Pipeline(name="DEITA")

load_data = LoadDataFromHub(
    name="load_data", batch_size=100, output_mappings={"prompt": "instruction"}, pipeline=pipeline
)
```

## EVOL-INSTRUCT: Generate Instructions with an LLM

[Evol-Instruct](https://arxiv.org/abs/2304.12244) automates the creation of complex instruction data for training large language models (LLMs) by progressively rewriting an initial set of instructions into more complex forms. This generated data is then used to fine-tune a model named WizardLM.

Evaluations show that instructions from Evol-Instruct are superior to human-created ones, and WizardLM achieves performance close to or exceeding GPT3.5-turbo in many skills. In distilabel, we initialise each step of the data generation pipeline. Later, we'll connect them together.

```python
evol_instruction_complexity = EvolInstruct(
    name="evol_instruction_complexity",
    llm=OpenAILLM(model="gpt-3.5-turbo"),
    num_evolutions=5,
    store_evolutions=True,
    generate_answers=True,
    include_original_instruction=True,
    pipeline=pipeline,
)

evol_instruction_complexity.load()

_evolved_instructions = next(evol_instruction_complexity.process(
    ([{"instruction": "How many fish are there in a dozen fish?"}]))
)

print(*_evolved_instructions, sep="\n")
```

Output:

```bash
( 1, 'How many fish are there in a dozen fish?')
( 2, 'How many rainbow trout are there in a dozen rainbow trout?')
( 3, 'What is the average weight in pounds of a dozen rainbow trout caught in a specific river in Alaska during the month of May?')
```

## EVOL COMPLEXITY: Evaluate complexity of generated instructions

The second step is the evaluation of *complexity* for an instruction in a given instruction-response pair. Like EVOL-INSTRUCT, this method uses LLMs instead of humans to automatically improve instructions, specifically through their complexity. From any instruction-response pair, $(I, R)$, we first generate new instructions following the In-Depth Evolving Response. We generate more complex instructions through prompting, as explained by authors, by adding some constraints or reasoning steps. Let\'s take an example from [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) which aims to generate observations by GPT-4 to instruct-tune LLMs with supervised fine-tuning. And, we have the instruction $instruction_0$:

```python
instruction_0 = "Give three tips for staying healthy."
```

To make it more complex, you can use, as the authors did, some prompt templates to add constraints or deepen the instruction. They provided some prompts in the paper appendix. For instance, this one was used to add constraints:

```python
PROMPT = """I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to
make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and
responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in
#Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
You SHOULD complicate the given prompt using the following method:
Please add one more constraints/requirements into #Given Prompt#
You should try your best not to make the #Rewritten Prompt# become verbose,
#Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’
are not allowed to appear in #Rewritten Prompt#
#Given Prompt#:
<Here is instruction>
#Rewritten Prompt#:
"""
```

Prompting this to an LLM, you automatically get a more complex instruction, called $instruction_1$, from an initial instruction $instruction_0$:

```python
instruction_1 = "Provide three recommendations for maintaining well-being, ensuring one focuses on mental health."
```

With sequences of evolved instructions, we use a further LLM to automatically rank and score them. We provide the 6 instructions at the same time. By providing all instructions together, we force the scoring model to look at minor complexity differences between evolved instructions. Encouraging the model to discriminate between instructions. Taking the example below, $instruction_0$ and $instruction_1$ could deserve the same score independently, but when compared together we would notice the slight difference that makes $instruction_1$ more complex.

In `distilabel`, we implement this like so:

```python
instruction_complexity_scorer = ComplexityScorer(
    name="instruction_complexity_scorer",
    llm=OpenAILLM(model="gpt-3.5-turbo"),
    input_mappings={"instructions": "evolved_instructions"},
    pipeline=pipeline,
)

expand_evolved_instructions = ExpandColumns(
    name="expand_evolved_instructions",
    columns=["evolved_instructions", "answers", "scores"],
    output_mappings={
        "evolved_instructions": "evolved_instruction",
        "answers": "answer",
        "scores": "evol_instruction_score",
    },
    pipeline=pipeline,
)

instruction_complexity_scorer.load()

_evolved_instructions = next(instruction_complexity_scorer.process(([{"evolved_instructions": [PROMPT + instruction_1]}])))

print("Original Instruction:")
print(instruction_1)
print("\nEvolved Instruction:")
print(_evolved_instructions[0]["evolved_instructions"][0].split("#Rewritten Prompt#:\n")[1])
```

Output:

```
Original Instruction:
Provide three recommendations for maintaining well-being, ensuring one focuses on mental health.

Evolved Instruction:
Suggest three strategies for nurturing overall well-being, with the stipulation that at least one explicitly addresses the enhancement of mental health, incorporating evidence-based practices.
```

## EVOL-QUALITY: Quality Evaluation

Now that we have scored the *complexity* of the instructions, we will focus on the *quality* of the responses. Similar to *EVOL COMPLEXITY*, the authors introduced *EVOL QUALITY, a method* based on LLMs, instead of humans, to automatically score the quality of the response.

From an instruction-response pair, $(I, R)$, the goal is to make the response evolve into a more helpful and relevant response. The key difference is that we need to also provide the first instruction to guide evolution. Let's take back our example from [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM).

Here we have the response $response_0$ and its initial instruction $instruction_0$:

```python
instruction_0 = "Give three tips for staying healthy."
reponse_0 = "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases. 2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week. 3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
```

Again the authors provided several prompts you could use to make your response evolve according to some guidelines. For example, this one was used to enrich the answer:

```python
PROMPT = """I want you to act as a Response Rewriter
Your goal is to enhance the quality of the response given by an AI assistant
to the #Given Prompt# through rewriting.
But the rewritten response must be reasonable and must be understood by humans.
Your rewriting cannot omit the non-text parts such as the table and code in
#Given Prompt# and #Given Response#. Also, please do not omit the input
in #Given Prompt#.
You Should enhance the quality of the response using the following method:
Please make the Response more in-depth
You should try your best not to make the #Rewritten Response# become verbose,
#Rewritten Response# can only add 10 to 20 words into #Given Response#.
‘#Given Response#’, ‘#Rewritten Response#’, ‘given response’ and ‘rewritten response’
are not allowed to appear in #Rewritten Response#
#Given Prompt#:
<instruction_0>
#Given Response#:
<response_0>
#Rewritten Response#:
"""
```

Prompting this to an LLM, you will automatically get a more enriched response, called $response_1$, from an initial response $response_0$ and initial instruction $instruction_0$:

```python
evol_response_quality = EvolQuality(
    name="evol_response_quality",
    llm=OpenAILLM(model="gpt-3.5-turbo"),
    num_evolutions=5,
    store_evolutions=True,
    include_original_response=True,
    input_mappings={
        "instruction": "evolved_instruction",
        "response": "answer",
    },
    pipeline=pipeline,
)

evol_response_quality.load()

_evolved_responses = next(evol_response_quality.process([{"instruction": PROMPT + instruction_0, "response": reponse_0}]))

print("Original Response:")
print(reponse_0)
print("\nEvolved Response:")
print(*_evolved_responses[0]['evolved_responses'], sep="\n")
```

And now, as in EVOL COMPLEXITY you iterate through this path and use different prompts to make your responses more relevant, helpful or creative. In the paper, they make 4 more iterations to get 5 evolved responses $(R0, R1, R2, R3, R4)$ which makes 5 different responses for one initial instruction at the end of this step.

```python
response_quality_scorer = QualityScorer(
    name="response_quality_scorer",
    llm=OpenAILLM(model="gpt-3.5-turbo"),
    input_mappings={
        "instruction": "evolved_instruction",
        "responses": "evolved_responses",
    },
    pipeline=pipeline,
)

expand_evolved_responses = ExpandColumns(
    name="expand_evolved_responses",
    columns=["evolved_responses", "scores"],
    output_mappings={
        "evolved_responses": "evolved_response",
        "scores": "evol_response_score",
    },
    pipeline=pipeline,
)

response_quality_scorer.load()

_scored_responses = next(response_quality_scorer.process([{"instruction": PROMPT + instruction_0, "responses": _evolved_responses[0]['evolved_responses']}]))

print("Original Response:")
print(reponse_0)

print("\nScore, Evolved Response:")
print(*zip(_scored_responses[0]["scores"], _evolved_responses[0]['evolved_responses']), sep="\n")
```

Output:

```bash
Original Response:
1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases. 2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week. 3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.

Score, Evolved Response:
(4.0, 'Here are three essential tips for maintaining good health: \n1. Prioritize regular exercise \n2. Eat a balanced diet with plenty of fruits and vegetables \n3. Get an adequate amount of sleep each night.')
(2.0, 'Here are three effective strategies to maintain a healthy lifestyle.')
(5.0, 'Here are three practical tips to maintain good health: Ensure a balanced diet, engage in regular exercise, and prioritize sufficient sleep. These practices support overall well-being.')
```

## Improving Data Diversity

One main component of good data to instruct-tune LLMs is diversity. Real world data can often contain [redundancy](https://openreview.net/forum?id=u96ZBg_Shna) due repetitive and homogeneous data.

The authors of the DEITA paper tackle the challenge of ensuring data diversity in the instruction tuning LLMs to avoid the pitfalls of data redundancy that can lead to over-fitting or poor generalization. They propose an embedding-based method to filter data for diversity. This method, called Repr Filter, uses embeddings generated by the *Llama 1 13B* model to represent instruction-response pairs in a vector space. The diversity of a new data sample is assessed based on the cosine distance between its embedding and that of its nearest neighbor in the already selected dataset. If this distance is greater than a specified threshold, the sample is considered diverse and is added to the selection. This process prioritizes diversity by assessing each sample's contribution to the variety of the dataset until the data selection budget is met. This approach effectively maintains the diversity of the data used for instruction tuning, as demonstrated by the DEITA models outperforming or matching state-of-the-art models with significantly less training data. In this implementation of DEITA we use the hidden state of the last layer of the Llama 2 model to generate embeddings, instead of a sentence transformer model, because we found that it improved the diversity of the data selection.

![jdjd](../../../assets/tutorials-assets/deita/diversity.png)

```python
generate_conversation = ConversationTemplate(
    name="generate_conversation",
    input_mappings={
        "instruction": "evolved_instruction",
        "response": "evolved_response",
    },
    pipeline=pipeline,
)

generate_embeddings = GenerateEmbeddings(
    name="generate_embeddings",
    llm=TransformersLLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cuda",
        torch_dtype="float16",
    ),
    input_mappings={"text": "conversation"},
    input_batch_size=5,
    pipeline=pipeline,
)

deita_filtering = DeitaFiltering(name="deita_filtering", pipeline=pipeline)
```

## Build the ⚗ distilabel `Pipeline`

Now we're ready to build a `distilabel` pipeline using the DEITA method:

```python
load_data.connect(evol_instruction_complexity)
evol_instruction_complexity.connect(instruction_complexity_scorer)
instruction_complexity_scorer.connect(expand_evolved_instructions)
expand_evolved_instructions.connect(evol_response_quality)
evol_response_quality.connect(response_quality_scorer)
response_quality_scorer.connect(expand_evolved_responses)
expand_evolved_responses.connect(generate_conversation)
generate_conversation.connect(generate_embeddings)
generate_embeddings.connect(deita_filtering)
```

Now we can run the pipeline. We use the step names to reference them in the pipeline configuration:

```python
distiset = pipeline.run(
    parameters={
        "load_data": {
            "repo_id": "distilabel-internal-testing/instruction-dataset-50",
            "split": "train",
        },
        "evol_instruction_complexity": {
            "llm": {"generation_kwargs": {"max_new_tokens": 512, "temperature": 0.7}}
        },
        "instruction_complexity_scorer": {
            "llm": {"generation_kwargs": {"temperature": 0.0}}
        },
        "evol_response_quality": {
            "llm": {"generation_kwargs": {"max_new_tokens": 512, "temperature": 0.7}}
        },
        "response_quality_scorer": {"llm": {"generation_kwargs": {"temperature": 0.0}}},
        "deita_filtering": {"data_budget": 500, "diversity_threshold": 0.04},
    },
    use_cache=False,
)
```

We can push the results to the Hugging Face Hub:

```python
distiset.push_to_hub("distilabel-internal-testing/deita-colab")
```

## Results

Again, to show the relevance of EVOL QUALITY method, the authors evaluated on the MT-bench models fine-tuned with different data selections according to how we defined quality responses according to an instruction. Each time they selected 6k data according to the quality score:

![DEITA results](../../../assets/tutorials-assets/deita/results.png)

Credit: Liu et al. (2023)

The score is much better when selecting data with the EVOL QUALITY method than when we select randomly or according to the length, making a more qualitative response if longer. Nevertheless, we see that the margin we may have seen in the complexity score is thinner. And we'll discuss the strategy in a later part. Nevertheless, this strategy looks to improve the fine-tuning compared to the baselines and now we're interested in mixing quality and complexity assessment with a diversity evaluation to find the right trade-off in our selection process.

## Conclusion

In conclusion, if you are looking for some efficient method to align an open-source LLM to your business case with a constrained budget, the solutions provided by DEITA are really worth the shot. This data-centric approach enables one to focus on the content of the dataset to have the best results instead of "just" scaling the instruction-tuning with more, and surely less qualitative, data. In a nutshell, the strategy developed, through automatically scoring instructions-responses, aims to substitute the human preference step proprietary models such as GPT-4 have been trained with. There are a few improvements we could think about when it comes to how to select the good data, but it opens a really great way in instruct-tuning LLM with lower computational needs making the whole process intellectually relevant and more sustainable than most of the other methods. We'd be happy to help you out with aligning an LLM with your business case drawing inspiration from such a methodology.
