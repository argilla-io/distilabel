<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/argilla-io/distilabel/blob/main/docs/assets/distilabel-white.png?raw=true">
    <img alt="Distilabel Logo" src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-black.png">
  </picture>
</div>

<h3 align="center">Synthesize data for AI and add feedback on the fly!</h3>

<p align="center">
  <a  href="https://pypi.org/project/distilabel/">
    <img alt="CI" src="https://img.shields.io/pypi/v/distilabel.svg?style=flat-round&logo=pypi&logoColor=white">
  </a>
  <a href="https://pepy.tech/project/distilabel">
    <img alt="CI" src="https://static.pepy.tech/personalized-badge/distilabel?period=month&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads/month">
  </a>
</p>

<p align="center">
  <a href="https://twitter.com/argilla_io">
    <img src="https://img.shields.io/badge/twitter-black?logo=x"/>
  </a>
  <a href="https://www.linkedin.com/company/argilla-io">
    <img src="https://img.shields.io/badge/linkedin-blue?logo=linkedin"/>
  </a>
  <a href="http://hf.co/join/discord">
  <img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
  </a>
</p>

Distilabel is the framework for synthetic data and AI feedback for engineers who need fast, reliable and scalable pipelines based on verified research papers.

If you just want to get started, we recommend you check the [documentation](http://distilabel.argilla.io/). Curious, and want to know more? Keep reading!
<!-- ![overview](https://github.com/argilla-io/distilabel/assets/36760800/360110da-809d-4e24-a29b-1a1a8bc4f9b7)  -->

## Why use distilabel?

Distilabel can be used for generating synthetic data and AI feedback for a wide variety of projects including traditional predictive NLP (classification, extraction, etc.), or generative and large language model scenarios (instruction following, dialogue generation, judging etc.). Distilabel's programmatic approach allows you to build scalable pipelines for data generation and AI feedback. The goal of distilabel is to accelerate your AI development by quickly generating high-quality, diverse datasets based on verified research methodologies for generating and judging with AI feedback.

### Improve your AI output quality through data quality

Compute is expensive and output quality is important. We help you **focus on data quality**, which tackles the root cause of both of these problems at once. Distilabel helps you to synthesize and judge data to let you spend your valuable time **achieving and keeping high-quality standards for your data**.

### Take control of your data and models

**Ownership of data for fine-tuning your own LLMs** is not easy but Distilabel can help you to get started. We integrate **AI feedback from any LLM provider out there** using one unified API.

### Improve efficiency by quickly iterating on the right research and LLMs

Synthesize and judge data with **latest research papers** while ensuring **flexibility, scalability and fault tolerance**. So you can focus on improving your data and training your models.

## Community

We are an open-source community-driven project and we love to hear from you. Here are some ways to get involved:

- [Community Meetup](https://lu.ma/embed-checkout/evt-IQtRiSuXZCIW6FB): listen in or present during one of our bi-weekly events.

- [Discord](http://hf.co/join/discord): get direct support from the community in #argilla-general and #argilla-help.

- [Roadmap](https://github.com/orgs/argilla-io/projects/10/views/1): plans change but we love to discuss those with our community so feel encouraged to participate.

## What do people build with Distilabel?

The Argilla community uses distilabel to create amazing [datasets](https://huggingface.co/datasets?other=distilabel) and [models](https://huggingface.co/models?other=distilabel).

- The [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) is a dataset of ~1 million AI preferences derived from teknium/OpenHermes-2.5. It shows how we can use Distilabel to **synthesize data on an immense scale**.
- Our [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) and the [improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B), show how we **improve model performance by filtering out 50%** of the original dataset through **AI feedback**.
- The [haiku DPO data](https://github.com/davanstrien/haiku-dpo) outlines how anyone can create a **dataset for a specific task** and **the latest research papers** to improve the quality of the dataset.

## Installation

```sh
pip install distilabel --upgrade
```

Requires Python 3.9+

In addition, the following extras are available:

### LLMs

- `anthropic`: for using models available in [Anthropic API](https://www.anthropic.com/api) via the `AnthropicLLM` integration.
- `cohere`: for using models available in [Cohere](https://cohere.ai/) via the `CohereLLM` integration.
- `argilla`: for exporting the generated datasets to [Argilla](https://argilla.io/).
- `groq`: for using models available in [Groq](https://groq.com/) using [`groq`](https://github.com/groq/groq-python) Python client via the `GroqLLM` integration.
- `hf-inference-endpoints`: for using the [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints) via the `InferenceEndpointsLLM` integration.
- `hf-transformers`: for using models available in [transformers](https://github.com/huggingface/transformers) package via the `TransformersLLM` integration.
- `litellm`: for using [`LiteLLM`](https://github.com/BerriAI/litellm) to call any LLM using OpenAI format via the `LiteLLM` integration.
- `llama-cpp`: for using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) Python bindings for `llama.cpp` via the `LlamaCppLLM` integration.
- `mistralai`: for using models available in [Mistral AI API](https://mistral.ai/news/la-plateforme/) via the `MistralAILLM` integration.
- `ollama`: for using [Ollama](https://ollama.com/) and their available models via `OllamaLLM` integration.
- `openai`: for using [OpenAI API](https://openai.com/blog/openai-api) models via the `OpenAILLM` integration, or the rest of the integrations based on OpenAI and relying on its client as `AnyscaleLLM`, `AzureOpenAILLM`, and `TogetherLLM`.
- `vertexai`: for using [Google Vertex AI](https://cloud.google.com/vertex-ai) proprietary models via the `VertexAILLM` integration.
- `vllm`: for using [vllm](https://github.com/vllm-project/vllm) serving engine via the `vLLM` integration.
- `sentence-transformers`: for generating sentence embeddings using [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
- `mlx`: for using [MLX](https://github.com/ml-explore/mlx) models via the `MlxLLM` integration.

### Structured generation

- `outlines`: for using structured generation of LLMs with [outlines](https://github.com/outlines-dev/outlines).
- `instructor`: for using structured generation of LLMs with [Instructor](https://github.com/jxnl/instructor/).

### Data processing

- `ray`: for scaling and distributing a pipeline with [Ray](https://github.com/ray-project/ray).
- `faiss-cpu` and `faiss-gpu`: for generating sentence embeddings using [faiss](https://github.com/facebookresearch/faiss).
- `text-clustering`: for using text clustering with [UMAP](https://github.com/lmcinnes/umap) and [Scikit-learn](https://github.com/scikit-learn/scikit-learn).
- `minhash`: for using minhash for duplicate detection with [datasketch](https://github.com/datasketch/datasketch) and [nltk](https://github.com/nltk/nltk).

### Example

To run the following example you must install `distilabel` with the `hf-inference-endpoints` extra:

```sh
pip install "distilabel[hf-inference-endpoints]" --upgrade
```

Then run:

```python
from datasets import load_dataset

from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            generation_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        ),
    )

if __name__ == "__main__":
    dataset = load_dataset("distilabel-internal-testing/instructions", split="test")
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="distilabel-example")
```

## Badges

If you build something cool with `distilabel` consider adding one of these badges to your dataset or model card.

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

## Contribute

To directly contribute with `distilabel`, check our [good first issues](https://github.com/argilla-io/distilabel/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [open a new one](https://github.com/argilla-io/distilabel/issues/new/choose).

## Improvements over Data Generation in RAG Tooling (data_gen)
- Modular pipeline system that allows for any type of generation (text, image, images, any output format) and composable steps whereas data_gen is sort of for questions and answers only (limited output format, no composability into more complex pipelines, not quite ready for multiple images).
    - This also makes it way more extensible. You can't really build on top of data_gen, only modify its internals to do some simple generation. You can build on top of this with 2 files, a config and a pipeline.
- Better parallelism by handling it with just a config and allowing pretty arbitrary gpu usage via tensor parallelism, replicas and available_gpus. data_gen only has the data parallelism wrapper I made which has no tensor parallelism support and requires sharding the chunks json manually before and after.
- Input and output in huggingface datasets rather than using the chunking library with its custom format and taking/outputting jsons.
- Built in and hidden caching for easy resuming
- According to the documentation, works with Ray for larger scale distributed generation.
- Inherits some cool things from distilabel such as the premade EvolInstructGenerator Task and [others](https://distilabel.argilla.io/latest/components-gallery/tasks/?h=task+gal).
- Slightly improved prompt sampler by making it part of the config (easier to edit and have multiple of) and adding the ability to generate list fields in an API call (say generate 4 questions instead of 1 and split these into separate rows)

## Notes About This Repo
- Run everything from the outside the `distilabel` directory. e.g. `python distilabel/pipelines/single_page_qa.py`
- In the modified distilabel package, here are some of the files I have added (you could also check the git commit history)
    - `pipelines/single_page_qa.py`. Put new pipelines here. The single page one is a good reference for how to do everything, copy and modify
    - `src/distilabel/configs/single_pages.py`. The config for single page QA, check it out to understand how the pipeline runs and what you can modify
    - `src/distilabel/pydantics.py`. Put Pydantic models here (configs, output formats)
    - `src/distilabel/llms/openai_compatible.py, vllm_api.py`. The wrapper that handles structured generation with openai compatible endpoints for different providers and vLLM servers as well.
    - `src/distilabel/utils/misc.py, prompt_sampler.py, pipe_utils.py, image.py`. Check out the prompt sampler and how it works in the config. `pipe_utils.py` has useful/reusable code for pipelines in general.
    - `src/distilabel/steps/columns/pydantic_to_cols.py`, `.../steps/filtering/filter_rows.py`, `.../steps/list_to_rows.py`, `.../tasks/lm_generation.py`. You can see each of them imported in the single page qa pipeline. `lm_generation.py` is important to know of because I use it for the structured generation step using a LM. Kind of obvious, but this is where your custom steps go.
- The only requirement for the dataset format is having a source column which is expected to be a string (straight input to the LM) or a list of image paths (which can point straight to jpg/png files or a page in a pdf with format `path/to/pdf_page_x.pdf`). This is atm only an expectation in `VLM._format_input()` when it is passed to `LMGenerationTask.input_formatter`, so you can change the `input_formatter`/override this if you need or just make `VLM._format_input()` more general.
- I handle scheduling gpus by overriding the available gpus seen by `CudaDevicePlacementMixin` and breaking the tasks into multiple load stages so that there are enough gpus available during each.
- It will launch a vllm server if the model name is not a recognized proprietary model.

## Notes on Distilabel (Issues and Helpful Knowledge)
- **Short Version**: distilabel is very particular about how things are done, so there's a reason why every line is the way it is and I recommend starting off of one of the existing pipelines. Also, reading my code for e.g. the single page pipeline will tell you how to build on top of distilabel. Use the rest of this list as an issue tracker so people know how to solve issues in the future.
---
- It took me a while to figure out how to handle different providers, it turns out their OpenAI compatible endpoints accept varying basic parameters and it works best to ignore most of the parameters and send basic messages.
- You can't output a pydantic object from a step since it isn't serializable with pyarrow.
- In as many places as possible, I think you want to use the `load()` method instead of `__init__`, since `__init__` is handled by pydantic and you'll be able to see the inherited args if you don't override it. It also matches the logic of the library better (matching load groups better for instance).
- I ran into some errors with the decorators that I tried to make for multiple generations and structured output because distilabel inspects the signature of functions and somehow came up with `**kwargs` was a required runtime parameter that needed to be set at pipeline init. The solution I am using is to copy the function signature from the library, though this isn't ideal for maintenance.
- I ran into some errors with it not being able to make a pyarrow table after finishing the `LMGenerationTask` which were due to the parameter `add_raw_input=True`. Since I overrode the `OpenAILLM` class to add support for more flexible vision (arbitrary number/order of images in chat format)(and to allow grouping all model providers under a single class), the formatted input was a list of messages, some text, some visual, all in one column (so you can vary the number of images). Pyarrow can't make a table out of this because the structure of a text and an image message are different so it can't make a type for the column. Thus, I have set `self.add_raw_input=False` in e.g. the `LMGenerationTask`.
    - This is no longer a current issue since I moved the prompt sampler into format input, which is called before the lm and discarded after (no serialization).
- `StepResources` seems like it might handle scheduling tasks across gpus for you, but I understand this only happens when using Ray, which has some internal scheduling that will respect those resources (there's a section in the documentation about how to use Ray for larger scale distributed generation). 
    - What it does actually do is respect `replicas`, which is basically just data parallelism for non-generator/global steps/tasks (replicates models as well).
    - It will put LLMs on different gpus (provided you use the mixin properly) until it runs out of gpus ([cuda_device_placement.py](distilabel/src/distilabel/models/mixins/cuda_device_placement.py)), but it won't reschedule them
- To handle scheduling tasks (say your pipeline will use 10 different vllm servers but you have 8 gpus), you use load stages. See the docs
- `Task.unload()` calls `self.llm.unload()` so you don't have to handle it yourself. If you wanted to keep it alive (say the vllm server), you'd need to get around this
- Distilabel can handle a list of tasks in the `>>` syntax, for each task in the previous stage, it sends the task's completed batches to all of the next stage (or in the case of using a router, it will select some set of the next stage per batch)
- Don't include a routing function step in the load groups, it isn't quite a step and will throw an error, but runs even when left out of the load groups
- I would like each LM to be able to have its own system prompt, which means they each need their own prompt sampler. I see two ways to do this, either make a step for each LM that has the prompt sampler and connect them properly, or put the prompt sampler with the LM. Making a bunch of steps and connecting them seems annoying and not as clean for writing new pipelines. Putting it with the LM means you don't see the system prompt since it isn't a step input or output, so I have sort of hacked distilabel by inplace updating input, which gets forwarded to `LMGenerationTask.format_output()`.
- Serialization
    - Initially, I ran into an error trying to hash my Config object (for the caching system) so I overrode the serialization to return an empty dict
    - When I was trying to test the caching, I ran into another error where it couldn't resume from the yaml because the `LMGenerationTask` has an input_formatter callable attribute. It loads the yaml with yaml.FullLoader which won't allow arbitrary python execution (setting the input_formatter). I found `Field(exclude=True)` in Pydantic to solve this. Then it occurred to me that I should do the same for the configs I was using rather than erasing their signatures. After this, there was another error in resuming because it couldn't initialize the e.g. `LMGenerationTask` without providing the configs. So, I gave these default initializations. This uncovered another error which was a bug in distilabel, I had no choice but to modify the actual package to fix it. In `DAG.from_dict()`, they don't set the `routing_batch_function._step` which is set during `Step.connect()`, so I just added the line to do that.
        - The way its resuming works is when you call `pipeline.run()`, one of the early steps is `self._refresh_pipeline_from_cache()` which essentially creates an entirely new dag from the cached information. Then, for excluded or secret fields, it sets them using the values of the current dag. Now that I know this, their design seems reasonable, but it is important that you understand the effect of `Field(exclude=True)` to get resuming working properly. The need for serialization and deserialization also justifies the extensive use of Pydantic in distilabel.
- Had to set `vllm_api` field to private so that it didn't try to serialize it in multiprocessing. 
- Might be errors with changing load_groups for a pipeline that you are trying to resume
- I made step resources an excluded parameter (from the signature and caching) so that you can change these and the pipeline will resume as normal
- [IMPORTANT] I ran into a tough error with distilabel hanging when trying to resume. The root cause (or one of them) was probably that I had stopped execution in the vscode debugger, which hard stops the program and distilabel didn't save the batch back to the pipeline's batch manager, making it so that my initial generator step didn't have its batch data and wasn't sending it up the pipeline. I am still not sure entirely how batches are routed, since this is a large and complex system, but anyways, be wary of the hanging issue. Keep in mind the functions `_manage_batch_flow, _BatchManagerStep._get_data(), get_batch() and add_batch() and _initialize_pipeline_execution()` which are related to batches in distilabel. I am not sure how exactly to solve this if it happens on something expensive to re-run. Maybe try manually editing the cache if you can find the right information. 


## Citation

```bibtex
@misc{distilabel-argilla-2024,
  author = {Álvaro Bartolomé Del Canto and Gabriel Martín Blázquez and Agustín Piqueres Lajarín and Daniel Vila Suero},
  title = {Distilabel: An AI Feedback (AIF) framework for building datasets with and for LLMs},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/argilla-io/distilabel}}
}
```
