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
import io
from typing import TYPE_CHECKING, Optional

from pydantic import validate_call

from distilabel.models.image_generation.base import AsyncImageGenerationModel
from distilabel.models.llms.huggingface import InferenceEndpointsLLM

if TYPE_CHECKING:
    from PIL import Image


class InferenceEndpointsImageGeneration(
    InferenceEndpointsLLM, AsyncImageGenerationModel
):
    """OpenAI image generation implementation running the async API client.

    Attributes:
        model_id: the model ID to use for the ImageGenerationModel as available in the Hugging Face Hub, which
            will be used to resolve the base URL for the serverless Inference Endpoints API requests.
            Defaults to `None`.
        endpoint_name: the name of the Inference Endpoint to use for the LLM. Defaults to `None`.
        endpoint_namespace: the namespace of the Inference Endpoint to use for the LLM. Defaults to `None`.
        base_url: the base URL to use for the Inference Endpoints API requests.
        api_key: the API key to authenticate the requests to the Inference Endpoints API.

    Icon:
        `:hugging:`

    Examples:
        Generate images from text prompts:

        ```python
        from distilabel.models.image_generation import InferenceEndpointsImageGeneration

        igm = InferenceEndpointsImageGeneration(model_id="black-forest-labs/FLUX.1-schnell", api_key="api.key")
        igm.load()

        output = igm.generate_outputs(
            inputs=["a white siamese cat"],
        )
        # [{"images": ["iVBORw0KGgoAAAANSUhEUgA..."]}]
        ```
    """

    @validate_call
    async def agenerate(
        self,
        input: str,
        negative_prompt: Optional[str] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        num_inference_steps: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_generations: int = 1,
    ) -> list[dict[str, any]]:
        """Generates images from text prompts using `huggingface_hub.AsyncInferenceClient.text_to_image`.

        Args:
            input: Prompt to generate an image from.
            negative_prompt: An optional negative prompt for the image generation. Defaults to None.
            height: The height in pixels of the image to generate.
            width: The width in pixels of the image to generate.
            num_inference_steps: The number of denoising steps. More denoising steps usually lead
                to a higher quality image at the expense of slower inference.
            guidance_scale: Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.

        Returns:
            A list with a dictionary containing a list with the image as a base64 string.
        """

        image: "Image" = await self._aclient.text_to_image(
            input,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return [{"images": [img_str]}]
