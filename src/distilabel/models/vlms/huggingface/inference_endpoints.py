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
from typing import TYPE_CHECKING, Any, Optional

from pydantic import validate_call

from distilabel.models.llms.huggingface import InferenceEndpointsLLM

if TYPE_CHECKING:
    from PIL import Image


class InferenceEndpointsImageLLM(InferenceEndpointsLLM):
    """OpenAI image generation implementation running the async API client.

    Attributes:
        model: the model name to use for the LLM e.g. "dall-e-3", etc.
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
        `:hugging:`

    Examples:
        Generate images from text prompts:

        ```python
        from distilabel.models.vlms import InferenceEndpointsImageLLM

        llm = InferenceEndpointsImageLLM(model="black-forest-labs/FLUX.1-schnell", api_key="api.key")
        llm.load()

        output = llm.generate_outputs(
            inputs=[{"prompt": "a white siamese cat"}],
        )
        # {"images": ["iVBORw0KGgoAAAANSUhEUgA..."]}
        ```
    """

    @validate_call
    async def agenerate(
        self,
        input: dict[str, Any],
        negative_prompt: Optional[str] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        num_inference_steps: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ) -> dict[str, list[str]]:
        """Generates images from text prompts using `huggingface_hub.AsyncInferenceClient.text_to_image`.

        Args:
            input: Input containing a dict with the key "prompt", with the prompt to generate an image from.
            negative_prompt: An optional negative prompt for the image generation. Defaults to None.
            height: The height in pixels of the image to generate.
            width: The width in pixels of the image to generate.
            num_inference_steps: The number of denoising steps. More denoising steps usually lead
                to a higher quality image at the expense of slower inference.
            guidance_scale: Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.

        Returns:
            A dictionary containing a list with the image as a base64 string.
        """

        image: "Image" = await self._aclient.text_to_image(
            input["prompt"],
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"images": [img_str]}
