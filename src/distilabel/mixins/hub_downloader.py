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

import os
import tempfile
from typing import Optional

from pydantic import BaseModel, Field


class HuggingFaceModelLoaderMixin(BaseModel):
    """
    A mixin for downloading models from the Hugging Face Hub.

    Attributes:
        repo_id (Optional[str]): The Hugging Face Hub repository id.
        model_file (str): The name of the model file to download.
        hf_token (Optional[str]): Hugging Face token for accessing gated models.
    """

    repo_id: Optional[str] = Field(
        default=None,
        description="The Hugging Face Hub repository id.",
    )
    model_file: str = Field(
        description="The name of the model file to download.",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for accessing gated models.",
    )

    def download_model(self) -> str:
        """
        Downloads the model from Hugging Face Hub if repo_id is provided.

        Returns:
            str: The path to the downloaded or local model file.

        Raises:
            ImportError: If huggingface_hub is not installed.
            ValueError: If repo_id is not provided or invalid.
            Exception: If there's an error downloading or loading the model.
        """
        if self.repo_id is None:
            return self.model_file

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import validate_repo_id
        except ImportError as ie:
            raise ImportError(
                "huggingface_hub package is not installed. "
                "You can install it with `pip install huggingface_hub`."
            ) from ie

        try:
            validate_repo_id(self.repo_id)
        except ValueError as ve:
            raise ValueError(f"Invalid repo_id: {self.repo_id}") from ve

        # Determine the download directory
        download_dir = os.environ.get("DISTILABEL_MODEL_DIR")
        if download_dir is None:
            download_dir = tempfile.gettempdir()

        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.model_file,
                token=self.hf_token,
                local_dir=download_dir,
            )
            return model_path
        except Exception as e:
            raise Exception(
                f"Failed to download model from Hugging Face Hub: {str(e)}"
            ) from e
