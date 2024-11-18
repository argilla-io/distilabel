---
hide:
  - navigation
---
# ImageGeneration

Image generation with an image to text model given a prompt.



`ImageGeneration` is a pre-defined task that allows generating images from a prompt.
    It works with any of the `image_generation` defined under `distilabel.models.image_generation`,
    the models implemented models that allow image generation.
    By default, the images are generated as a base64 string format, and after the dataset
    has been generated, the images can be automatically transformed to `PIL.Image.Image` using
    `Distiset.transform_columns_to_image`. Take a look at the `Image Generation with distilabel`
    example in the documentation for more information.
    Using the `save_artifacts` attribute, the images can be saved on the artifacts folder in the
    hugging face hub repository.





### Attributes

- **save_artifacts**: Bool value to save the image artifacts on its folder.  Otherwise, the base64 representation of the image will be saved as  a string. Defaults to False.

- **image_format**: Any of the formats supported by PIL. Defaults to `JPEG`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[prompt]
		end
		subgraph New columns
			OCOL0[image]
			OCOL1[image_path]
			OCOL2[model_name]
		end
	end

	subgraph ImageGeneration
		StepInput[Input Columns: prompt]
		StepOutput[Output Columns: image, image_path, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **prompt** (str): A column named prompt with the prompts to generate the images.




#### Outputs


- **image** (`str`): The generated image. Initially is a base64 string, for simplicity  during the .

- **image_path** (`str`): The path where the image is saved. Only available if `save_artifacts`  is True.

- **model_name** (`str`): The name of the model used to generate the image.





### Examples


#### Generate an image from a prompt
```python
from distilabel.steps.tasks import ImageGeneration
# Select the Image Generation model to use
from distilabel.models.image_generation import OpenAIImageGeneration
from distilabel.models.image_generation import InferenceEndpointsImageGeneration

ilm = InferenceEndpointsImageGeneration(
    model_id="black-forest-labs/FLUX.1-schnell"
)
ilm = OpenAIImageGeneration(
    model="dall-e-3",
    api_key="api.key",
    generation_kwargs={
        "size": "1024x1024",
        "quality": "standard",
        "style": "natural"
    }
)

# save_artifacts=True by default in JPEG format, if set to False, the image will be saved as a string.
image_gen = ImageGeneration(
    llm=ilm,
    save_artifacts=True,
    image_format="JPEG"
)

image_gen.load()

result = next(
    image_gen.process(
        [{"prompt": "a white siamese cat"}]
    )
)
```




