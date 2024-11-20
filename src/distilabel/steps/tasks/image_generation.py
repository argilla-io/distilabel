# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
from typing import TYPE_CHECKING

from distilabel.models.image_generation.utils import image_from_str
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import ImageTask

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class ImageGeneration(ImageTask):
    """Image generation with an image to text model given a prompt.

    `ImageGeneration` is a pre-defined task that allows generating images from a prompt.
    It works with any of the `image_generation` defined under `distilabel.models.image_generation`,
    the models implemented models that allow image generation.
    By default, the images are generated as a base64 string format, and after the dataset
    has been generated, the images can be automatically transformed to `PIL.Image.Image` using
    `Distiset.transform_columns_to_image`. Take a look at the `Image Generation with distilabel`
    example in the documentation for more information.
    Using the `save_artifacts` attribute, the images can be saved on the artifacts folder in the
    hugging face hub repository.

    Attributes:
        save_artifacts: Bool value to save the image artifacts on its folder.
            Otherwise, the base64 representation of the image will be saved as
            a string. Defaults to False.
        image_format: Any of the formats supported by PIL. Defaults to `JPEG`.

    Input columns:
        - prompt (str): A column named prompt with the prompts to generate the images.

    Output columns:
        - image (`str`): The generated image. Initially is a base64 string, for simplicity
            during the pipeline run, but this can be transformed to an Image object after
            distiset is returned at the end of a pipeline by calling
            `distiset.transform_columns_to_image(<IMAGE_COLUMN>)`.
        - image_path (`str`): The path where the image is saved. Only available if `save_artifacts`
            is True.
        - model_name (`str`): The name of the model used to generate the image.

    Categories:
        - image-generation

    Examples:
        Generate an image from a prompt:

        ```python
        from distilabel.steps.tasks import ImageGeneration
        from distilabel.models.image_generation import InferenceEndpointsImageGeneration

        igm = InferenceEndpointsImageGeneration(
            model_id="black-forest-labs/FLUX.1-schnell"
        )

        # save_artifacts=True by default in JPEG format, if set to False, the image will be saved as a string.
        image_gen = ImageGeneration(image_generation_model=igm)

        image_gen.load()

        result = next(
            image_gen.process(
                [{"prompt": "a white siamese cat"}]
            )
        )
        ```

        Generate an image and save them as artifacts in a Hugging Face Hub repository:

        ```python
        from distilabel.steps.tasks import ImageGeneration
        # Select the Image Generation model to use
        from distilabel.models.image_generation import OpenAIImageGeneration

        igm = OpenAIImageGeneration(
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
            image_generation_model=igm,
            save_artifacts=True,
            image_format="JPEG"  # By default will use JPEG, the options available can be seen in PIL documentation.
        )

        image_gen.load()

        result = next(
            image_gen.process(
                [{"prompt": "a white siamese cat"}]
            )
        )
        ```
    """

    save_artifacts: bool = False
    image_format: str = "JPEG"

    @property
    def inputs(self) -> "StepColumns":
        return ["prompt"]

    @property
    def outputs(self) -> "StepColumns":
        return {
            "image": True,
            "image_path": False,
            "model_name": True,
        }

    def format_input(self, input: dict[str, any]) -> str:
        return input["prompt"]

    def format_output(
        self, output: dict[str, any], input: dict[str, any]
    ) -> dict[str, any]:
        image = None
        if img_str := output.get("images"):
            image = img_str[0]  # Grab only the first image

        return {"image": image, "model_name": self.llm.model_name}

    def save(self, **kwargs):
        if not self.save_artifacts:
            from distilabel.utils.serialization import _Serializable

            super(_Serializable).save(**kwargs)

    def process(self, inputs: StepInput) -> "StepOutput":
        formatted_inputs = self._format_inputs(inputs)

        outputs = self.llm.generate_outputs(
            inputs=formatted_inputs,
            num_generations=self.num_generations,
            **self.llm.get_generation_kwargs(),
        )

        task_outputs = []
        for input, input_outputs in zip(inputs, outputs):
            formatted_outputs = self._format_outputs(input_outputs, input)
            for formatted_output in formatted_outputs:
                if self.save_artifacts and (
                    image := formatted_output.get("image", None)
                ):
                    # use prompt as filename
                    prompt_hash = hashlib.md5(input["prompt"].encode()).hexdigest()
                    # Build PIL image to save it
                    image = image_from_str(image)

                    self.save_artifact(
                        name="images",
                        write_function=lambda path,
                        prompt_hash=prompt_hash,
                        img=image: img.save(
                            path / f"{prompt_hash}.{self.image_format.lower()}",
                            format=self.image_format,
                        ),
                        metadata={"type": "image"},
                    )
                    formatted_output["image_path"] = (
                        f"artifacts/{self.name}/images/{prompt_hash}.{self.image_format.lower()}"
                    )

                task_outputs.append(
                    {**input, **formatted_output, "model_name": self.llm.model_name}
                )
        yield task_outputs
