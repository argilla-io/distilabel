---
hide:
  - navigation
---
# ImageGeneration

Image generation with a Vision Language Model (VLM) given a prompt.



`ImageGeneration` is a pre-defined task that allows generating images from a prompt.
    It works with any of the `vlms` defined under `distilabel.models.vlms`, the models
    implemented models that allow image generation.
    By default, the images are saved as JPEG files, but this can be changed using the
    `save_artifacts` and `image_format` attributes.





### Attributes

- **save_artifacts**: Bool value to save the image artifacts on its folder.  Otherwise, the base64 representation of the image will be saved as  a string. Defaults to True.

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
			OCOL1[model_name]
		end
	end

	subgraph ImageGeneration
		StepInput[Input Columns: prompt]
		StepOutput[Output Columns: image, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **prompt** (str): A column named prompt with the prompts to generate the images.




#### Outputs


- **image** (`str`): The generated image.

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




