# How-to guides

Welcome to the how-to guides section! Here you will find a collection of guides that will help you get started with Distilabel. We have divided the guides into two categories: basic and advanced. The basic guides will help you get started with the core concepts of Distilabel, while the advanced guides will help you explore more advanced features.

## Basic

<div class="grid cards" markdown>

-   __Define Steps for your Pipeline__

    ---

    Steps are the building blocks of your pipeline. They can be used to generate data, evaluate models, manipulate data, or any other general task.

    [:octicons-arrow-right-24: Define Steps](basic/step/index.md)

-   __Define Tasks that rely on LLMs__

    ---

    Tasks are a specific type of step that rely on Language Models (LLMs) to generate data.

    [:octicons-arrow-right-24: Define Tasks](basic/task/index.md)

-   __Define LLMs as local or remote models__

    ---

    LLMs are the core of your tasks. They are used to integrate with local models or remote APIs.

    [:octicons-arrow-right-24: Define LLMs](basic/llm/index.md)

-   __Execute Steps and Tasks in a Pipeline__

    ---

    Pipeline is where you put all your steps and tasks together to create a workflow.

    [:octicons-arrow-right-24: Execute Pipeline](basic/pipeline/index.md)

</div>

## Advanced

<div class="grid cards" markdown>
-  __Using the Distiset dataset object__

    ---

    Distiset is a dataset object based on the datasets library that can be used to store and manipulate data.

    [:octicons-arrow-right-24: Distiset](advanced/distiset.md)

-  __Export data to Argilla__

    ---

    Argilla is a platform that can be used to store, search, and apply feedback to datasets.
    [:octicons-arrow-right-24: Argilla](advanced/argilla.md)

-  __Using a file system to pass data of batches between steps__

    ---

    File system can be used to pass data between steps in a pipeline.

    [:octicons-arrow-right-24: File System](advanced/fs_to_pass_data.md)

-  __Using CLI to explore and re-run existing Pipelines__

    ---

    CLI can be used to explore and re-run existing pipelines through the command line.

    [:octicons-arrow-right-24: CLI](advanced/cli/index.md)

-  __Cache and recover pipeline executions__

    ---

    Caching can be used to recover pipeline executions to avoid loosing data and precious LLM calls.

    [:octicons-arrow-right-24: Caching](advanced/caching.md)

-  __Structured data generation__

    ---

    Structured data generation can be used to generate data with a specific structure like JSON, function calls, etc.

    [:octicons-arrow-right-24: Structured Generation](advanced/structured_generation.md)

-  __Serving an LLM for sharing it between several tasks__

    ---

    Serve an LLM via TGI or vLLM to make requests and connect using a client like `InferenceEndpointsLLM` or `OpenAILLM` to avoid wasting resources.

    [:octicons-arrow-right-24: Sharing an LLM across tasks](advanced/serving_an_llm_for_reuse.md)

-  __Impose requirements to your pipelines and steps__

    ---

    Add requirements to steps in a pipeline to ensure they are installed and avoid errors.

    [:octicons-arrow-right-24: Pipeline requirements](advanced/pipeline_requirements.md)

</div>