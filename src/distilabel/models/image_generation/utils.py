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

from PIL import Image


def image_to_str(image: Image.Image, image_format: str = "JPEG") -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_from_str(image_str: str) -> Image.Image:
    """Converts a base64 encoded string to a PIL Image."""
    image_bytes = base64.b64decode(image_str)
    return Image.open(io.BytesIO(image_bytes))
