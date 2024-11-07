# Serving an `LLM` for sharing it between several `Task`s

It's very common to want to use the same `LLM` for several `Task`s in a pipeline. To avoid loading the `LLM` as many times as the number of `Task`s and avoid wasting resources, it's recommended to serve the model using solutions like [`text-generation-inference`](https://huggingface.co/docs/text-generation-inference/quicktour#launching-tgi) or [`vLLM`](https://docs.vllm.ai/en/stable/serving/deploying_with_docker.html), and then use an `AsyncLLM` compatible client like `InferenceEndpointsLLM` or `OpenAILLM` to communicate with the server respectively.

## Serving LLMs using `text-generation-inference`

```bash
model=meta-llama/Meta-Llama-3-8B-Instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    -e HUGGING_FACE_HUB_TOKEN=<secret> \
    ghcr.io/huggingface/text-generation-inference:2.0.4 \
    --model-id $model
```

!!! NOTE

    The bash command above has been copy-pasted from the official docs [text-generation-inference](https://huggingface.co/docs/text-generation-inference/quicktour#launching-tgi). Please refer to the official docs for more information.

And then we can use `InferenceEndpointsLLM` with `base_url=http://localhost:8080` (pointing to our `TGI` local deployment):

```python
from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration, UltraFeedback

with Pipeline(name="serving-llm") as pipeline:
    load_data = LoadDataFromDicts(
        data=[{"instruction": "Write a poem about the sun and moon."}]
    )

    # `base_url` points to the address of the `TGI` serving the LLM
    llm = InferenceEndpointsLLM(base_url="http://192.168.1.138:8080")

    text_generation = TextGeneration(
        llm=llm,
        num_generations=3,
        group_generations=True,
        output_mappings={"generation": "generations"},
    )

    ultrafeedback = UltraFeedback(aspect="overall-rating", llm=llm)

    load_data >> text_generation >> ultrafeedback
```


## Serving LLMs using `vLLM`

```bash
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

!!! NOTE

    The bash command above has been copy-pasted from the official docs [vLLM](https://docs.vllm.ai/en/stable/serving/deploying_with_docker.html). Please refer to the official docs for more information.

And then we can use `OpenAILLM` with `base_url=http://localhost:8000` (pointing to our `vLLM` local deployment):

```python
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration, UltraFeedback

with Pipeline(name="serving-llm") as pipeline:
    load_data = LoadDataFromDicts(
        data=[{"instruction": "Write a poem about the sun and moon."}]
    )

    # `base_url` points to the address of the `vLLM` serving the LLM
    llm = OpenAILLM(base_url="http://192.168.1.138:8000", model="")

    text_generation = TextGeneration(
        llm=llm,
        num_generations=3,
        group_generations=True,
        output_mappings={"generation": "generations"},
    )

    ultrafeedback = UltraFeedback(aspect="overall-rating", llm=llm)

    load_data >> text_generation >> ultrafeedback
```
