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

import base64
from typing import TYPE_CHECKING, Literal, Optional

import requests
from pydantic import validate_call

from distilabel.models.image_generation.base import AsyncImageGenerationModel
from distilabel.models.llms.openai import OpenAILLM

if TYPE_CHECKING:
    from openai.types import ImagesResponse


class OpenAIImageGeneration(AsyncImageGenerationModel, OpenAILLM):
    """OpenAI image generation implementation running the async API client.

    Attributes:
        model: the model name to use for the ImageGenerationModel e.g. "dall-e-3", etc.
            Supported models can be found [here](https://platform.openai.com/docs/guides/images).
        base_url: the base URL to use for the OpenAI API requests. Defaults to `None`, which
            means that the value set for the environment variable `OPENAI_BASE_URL` will
            be used, or "https://api.openai.com/v1" if not set.
        api_key: the API key to authenticate the requests to the OpenAI API. Defaults to
            `None` which means that the value set for the environment variable `OPENAI_API_KEY`
            will be used, or `None` if not set.
        max_retries: the maximum number of times to retry the request to the API before
            failing. Defaults to `6`.
        timeout: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.

    Icon:
        `:simple-openai:`

    Examples:
        Generate images from text prompts:

        ```python
        from distilabel.models.image_generation import OpenAIImageGeneration

        igm = OpenAIImageGeneration(model="dall-e-3", api_key="api.key")

        igm.load()

        output = igm.generate_outputs(
            inputs=["a white siamese cat"],
            size="1024x1024",
            quality="standard",
            style="natural",
        )
        # [{"images": ["iVBORw0KGgoAAAANSUhEUgA..."]}]
        ```
    """

    @property
    def model_name(self) -> str:
        return OpenAILLM.model_name.fget(self)

    def load(self) -> None:
        OpenAILLM.load(self)

    def unload(self) -> None:
        OpenAILLM.unload(self)

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: str,
        num_generations: int = 1,
        quality: Optional[Literal["standard", "hd"]] = "standard",
        response_format: Optional[Literal["url", "b64_json"]] = "url",
        size: Optional[
            Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
        ] = None,
        style: Optional[Literal["vivid", "natural"]] = None,
    ) -> list[dict[str, any]]:
        """Generates `num_generations` images for the given input using the OpenAI async
        client. The images are base64 string representations.

        Args:
            input: A text description of the desired image(s). The maximum length is 1000
                characters for `dall-e-2` and 4000 characters for `dall-e-3`.
            num_generations: The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only
                `n=1` is supported.
            quality: The quality of the image that will be generated. `hd` creates images with finer
                details and greater consistency across the image. This param is only supported
                for `dall-e-3`.
            response_format: The format in which the generated images are returned. Must be one of `url` or
                `b64_json`. URLs are only valid for 60 minutes after the image has been
                generated.
            size: The size of the generated images. Must be one of `256x256`, `512x512`, or
                `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or
                `1024x1792` for `dall-e-3` models.
            style: The style of the generated images. Must be one of `vivid` or `natural`. Vivid
                causes the model to lean towards generating hyper-real and dramatic images.
                Natural causes the model to produce more natural, less hyper-real looking
                images. This param is only supported for `dall-e-3`.

        Returns:
            A list with a dictionary with the list of images generated.
        """
        images_response: "ImagesResponse" = await self._aclient.images.generate(
            model=self.model_name,
            prompt=input,
            n=num_generations,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
        )
        images = []
        for image in images_response.data:
            if response_format == "url":
                image_data = requests.get(
                    image.url
                ).content  # TODO: Keep a requests/httpx session instead
                image_str = base64.b64encode(image_data).decode()
                images.append(image_str)
            elif response_format == "b64_json":
                images.append(image.b64_json)
        return [{"images": images}]
