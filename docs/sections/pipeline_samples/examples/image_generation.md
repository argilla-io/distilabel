---
hide: toc
---

# Image generation with `distilabel`

Create synthetic images using `distilabel`.

This example shows how distilabel can be used to generate image data, either using [`InferenceEndpointsImageLM`](https://distilabel.argilla.io/dev/components-gallery/image_generation/inferenceendpointsimagelm/) or [`OpenAIImageLM`](https://distilabel.argilla.io/dev/components-gallery/image_generation/openaiimagelm/).


=== "Inference Endpoints - black-forest-labs/FLUX.1-schnell"

    ```python
    from distilabel.pipeline import Pipeline
    from distilabel.steps import KeepColumns
    from distilabel.models.image_generation import InferenceEndpointsImageLM
    from distilabel.steps.tasks import ImageGeneration

    from datasets import load_dataset

    ds = load_dataset("dvilasuero/finepersonas-v0.1-tiny", split="train").select(range(3))

    with Pipeline(name="image_generation_pipeline") as pipeline:
        ilm = InferenceEndpointsImageLM(
            model_id="black-forest-labs/FLUX.1-schnell"
        )

        img_generation = ImageGeneration(
            name="flux_schnell",
            llm=ilm,
            input_mappings={"prompt": "persona"}
        )
        
        keep_columns = KeepColumns(columns=["persona", "model_name", "image"])

        img_generation >> keep_columns
    ```

    Sample image for the prompt:

    > A local art historian and museum professional interested in 19th-century American art and the local cultural heritage of Cincinnati.

    ![image_ie](https://huggingface.co/datasets/plaguss/test-finepersonas-v0.1-tiny-flux-schnell/resolve/main/artifacts/flux_schnell/images/3333f9870feda32a449994017eb72675.jpeg)

=== "OpenAI - dall-e-3"

    ```python
    from distilabel.pipeline import Pipeline
    from distilabel.steps import KeepColumns
    from distilabel.models.image_generation import OpenAIImageLM
    from distilabel.steps.tasks import ImageGeneration

    from datasets import load_dataset

    ds = load_dataset("dvilasuero/finepersonas-v0.1-tiny", split="train").select(range(3))

    with Pipeline(name="image_generation_pipeline") as pipeline:
        ilm = OpenAIImageLM(
            model="dall-e-3",
            generation_kwargs={
                "size": "1024x1024",
                "quality": "standard",
                "style": "natural"
            }
        )

        img_generation = ImageGeneration(
            name="dalle-3"
            llm=ilm,
            input_mappings={"prompt": "persona"}
        )
        
        keep_columns = KeepColumns(columns=["persona", "model_name", "image"])

        img_generation >> keep_columns
    ```

    Sample image for the prompt:

    > A local art historian and museum professional interested in 19th-century American art and the local cultural heritage of Cincinnati.

    ![image_oai](https://huggingface.co/datasets/plaguss/test-finepersonas-v0.1-tiny-dall-e-3/resolve/main/artifacts/dalle-3/images/3333f9870feda32a449994017eb72675.jpeg)

The full pipeline can be run at the following example. Keep in mind, you need to install `pillow` first: `pip install distilabel[vision]`.

??? Run

    ```python
    python examples/image_generation.py
    ```

```python title="image_generation.py"
--8<-- "examples/image_generation.py"
```
