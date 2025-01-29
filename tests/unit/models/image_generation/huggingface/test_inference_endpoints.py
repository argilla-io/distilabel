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


from unittest.mock import AsyncMock, MagicMock, patch

import nest_asyncio
import numpy as np
import pytest
from PIL import Image

from distilabel.models.image_generation.huggingface.inference_endpoints import (
    InferenceEndpointsImageGeneration,
)

return_value = MagicMock()
return_value.get_model_status.return_value = MagicMock(
    state="Loaded", framework="text-generation-inference"
)
return_value._resolve_url.return_value = "http://localhost:8000"


@patch("huggingface_hub.AsyncInferenceClient", return_value=return_value)
@patch("huggingface_hub.InferenceClient", return_value=return_value)
class TestInferenceEndpointsImageGeneration:
    @pytest.mark.asyncio
    async def test_agenerate(self, mock_inference_client: MagicMock) -> None:
        igm = InferenceEndpointsImageGeneration(
            model_id="black-forest-labs/FLUX.1-schnell",
            api_key="api.key",
        )
        igm.load()

        arr = np.random.randint(0, 255, (100, 100, 3))
        random_image = Image.fromarray(arr, "RGB")
        igm._aclient.text_to_image = AsyncMock(return_value=random_image)

        assert await igm.agenerate("Aenean hend")

    @pytest.mark.asyncio
    async def test_generate(self, mock_inference_client: MagicMock) -> None:
        igm = InferenceEndpointsImageGeneration(
            model_id="black-forest-labs/FLUX.1-schnell",
            api_key="api.key",
        )
        igm.load()

        arr = np.random.randint(0, 255, (100, 100, 3))
        random_image = Image.fromarray(arr, "RGB")
        igm._aclient.text_to_image = AsyncMock(return_value=random_image)

        nest_asyncio.apply()

        images = igm.generate(inputs=["Aenean hendrerit aliquam velit. ..."])
        assert images[0][0]["images"][0].startswith("/9j/4AAQSkZJRgABAQAAAQABAAD/2w")
