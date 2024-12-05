---
hide: toc
---

# Vision generation with `distilabel`

Answer questions about images using `distilabel`.

Image-text-to-text models take in an image and text prompt and output text. In this example we will use an LLM [`InferenceEndpointsLLM`](https://distilabel.argilla.io/dev/components-gallery/llms/inferenceendpointsllm/) with [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) to ask a question about an image, and [`OpenAILLM`](https://distilabel.argilla.io/dev/components-gallery/llms/openaillm/) with `gpt-4o-mini`. We will ask a simple question to showcase how the [`TextGenerationWithImage`](https://distilabel.argilla.io/dev/components-gallery/tasks/textgenerationwithimage/) task can be used in a pipeline.

=== "Inference Endpoints - meta-llama/Llama-3.2-11B-Vision-Instruct"

    ```python
    from distilabel.models.llms import InferenceEndpointsLLM
    from distilabel.pipeline import Pipeline
    from distilabel.steps.tasks.text_generation_with_image import TextGenerationWithImage
    from distilabel.steps import LoadDataFromDicts


    with Pipeline(name="vision_generation_pipeline") as pipeline:
        loader = LoadDataFromDicts(
            data=[
                {
                    "instruction": "What’s in this image?",
                    "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            ],
        )

        llm = InferenceEndpointsLLM(
            model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        )

        vision = TextGenerationWithImage(
            name="vision_gen",
            llm=llm,
            image_type="url"  # (1)
        )

        loader >> vision
    ```

    1. The *image_type* can be a url pointing to the image, the base64 string representation, or a PIL image, take a look at the [`TextGenerationWithImage`](https://distilabel.argilla.io/dev/components-gallery/tasks/textgenerationwithimage/) for more information.

    Image:

    ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg)

    Question:

    > What’s in this image?

    Response:

    > This image depicts a wooden boardwalk weaving its way through a lush meadow, flanked by vibrant green grass that stretches towards the horizon under a calm and inviting sky. The boardwalk runs straight ahead, away from the viewer, forming a clear pathway through the tall, lush green grass, crops or other plant types or an assortment of small trees and shrubs. This meadow is dotted with trees and shrubs, appearing to be healthy and green. The sky above is a beautiful blue with white clouds scattered throughout, adding a sense of tranquility to the scene. While this image appears to be of a natural landscape, because grass is...

=== "OpenAI - gpt-4o-mini"

    ```python
    from distilabel.models.llms import OpenAILLM
    from distilabel.pipeline import Pipeline
    from distilabel.steps.tasks.text_generation_with_image import TextGenerationWithImage
    from distilabel.steps import LoadDataFromDicts


    with Pipeline(name="vision_generation_pipeline") as pipeline:
        loader = LoadDataFromDicts(
            data=[
                {
                    "instruction": "What’s in this image?",
                    "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            ],
        )

        llm = OpenAILLM(
            model="gpt-4o-mini",
        )

        vision = TextGenerationWithImage(
            name="vision_gen",
            llm=llm,
            image_type="url"  # (1)
        )

        loader >> vision
    ```

    1. The *image_type* can be a url pointing to the image, the base64 string representation, or a PIL image, take a look at the [`VisionGeneration`](https://distilabel.argilla.io/dev/components-gallery/tasks/visiongeneration/) for more information.

    Image:

    ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg)

    Question:

    > What’s in this image?

    Response:

    > The image depicts a scenic landscape featuring a wooden walkway or path that runs through a lush green marsh or field. The area is surrounded by tall grass and various shrubs, with trees likely visible in the background. The sky is blue with some wispy clouds, suggesting a beautiful day. Overall, it presents a peaceful natural setting, ideal for a stroll or nature observation.


The full pipeline can be run at the following example:

??? Note "Run the full pipeline"

    ```python
    python examples/text_generation_with_image.py
    ```

    ```python title="text_generation_with_image.py"
    --8<-- "examples/text_generation_with_image.py"
    ```

A sample dataset can be seen at [plaguss/test-vision-generation-Llama-3.2-11B-Vision-Instruct](https://huggingface.co/datasets/plaguss/test-vision-generation-Llama-3.2-11B-Vision-Instruct).
