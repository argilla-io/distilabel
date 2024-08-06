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
from pathlib import Path
from typing import Final

from huggingface_hub import constants

HF_TOKEN_ENV_VAR: Final[str] = "HF_TOKEN"


def get_hf_token(cls_name: str, token_arg: str) -> str:
    """Get the token for the hugging face API.

    Tries to extract it from the environment variable, if it is not found
    it tries to read it from the file using 'huggingface_hub',
    and if not possible raises a ValueError.

    Args:
        cls_name: Name of the class/function that requires the token.
        token_arg: Argument name to use in the error message, normally
            is "token" or "api_key".

    Raises:
        ValueError: If the token is not found in the file.

    Returns:
        The token for the hugging face API.
    """
    token = os.getenv(HF_TOKEN_ENV_VAR)
    if token is None:
        if not Path(constants.HF_TOKEN_PATH).exists():
            raise ValueError(
                f"To use `{cls_name}` an API key must be provided via `{token_arg}`,"
                f" set the environment variable `{HF_TOKEN_ENV_VAR}` or use the"
                " `huggingface-hub` CLI to login with `huggingface-cli login`."
            )
        with open(constants.HF_TOKEN_PATH) as f:
            token = f.read().strip()
    return token
