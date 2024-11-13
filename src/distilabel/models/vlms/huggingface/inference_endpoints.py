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
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from pydantic import validate_call

from distilabel.models.llms.huggingface import InferenceEndpointsLLM

if TYPE_CHECKING:
    from PIL import Image


class InferenceEndpointsImageLLM(InferenceEndpointsLLM):
    @validate_call
    async def agenerate(
        self,
        input: dict[str, any],
        negative_prompt: Optional[str] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        num_inference_steps: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ) -> list[dict[str, any]]:
        """Generates images from text prompts using `huggingface_hub.AsyncClient.text_to_image`.

        Args:
            input: Input containing a dict with the key "prompt", with.
                the prompt to generate an image from.
            negative_prompt: An optional negative prompt for the image generation. Defaults to None.
            height: The height in pixels of the image to generate.
            width: The width in pixels of the image to generate.
            num_inference_steps: The number of denoising steps. More denoising steps usually lead
                to a higher quality image at the expense of slower inference.
            guidance_scale: Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.

        Returns:
            TODO: SHOULD BE THE IMAGE, OR A LIST WITH DICT AND THE IMAGE THERE?
        """

        prompt = input.get("prompt", "")
        image: "Image" = await self._aclient.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return [{"image": img_str}]


class InferenceEndpointsImageToImageLLM(InferenceEndpointsLLM):
    @validate_call
    async def agenerate(
        self,
        input: dict[str, any],
        negative_prompt: Optional[str] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        num_inference_steps: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ) -> list[dict[str, any]]:
        """Generates images from text prompts using `huggingface_hub.AsyncClient.text_to_image`.

        Args:
            input: Input containing a dict with the key "prompt", with.
                the prompt to generate an image from.
            negative_prompt: An optional negative prompt for the image generation. Defaults to None.
            height: The height in pixels of the image to generate.
            width: The width in pixels of the image to generate.
            num_inference_steps: The number of denoising steps. More denoising steps usually lead
                to a higher quality image at the expense of slower inference.
            guidance_scale: Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.

        Returns:
            TODO: SHOULD BE THE IMAGE, OR A LIST WITH DICT AND THE IMAGE THERE?
        """

        prompt = input.get("prompt", "")
        image: "Image" = await self._aclient.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return [{"image": img_str}]
