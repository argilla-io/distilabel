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
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
)

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.typing import StructuredOutputType
from distilabel.utils.huggingface import HF_TOKEN_ENV_VAR, get_hf_token

if TYPE_CHECKING:
    from huggingface_hub import AsyncInferenceClient
    from transformers import PreTrainedTokenizer


class InferenceEndpointsBaseClient(BaseModel):
    model_id: Optional[str] = None

    endpoint_name: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The name of the Inference Endpoint to use for the LLM.",
    )
    endpoint_namespace: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The namespace of the Inference Endpoint to use for the LLM.",
    )
    base_url: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The base URL to use for the Inference Endpoints API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(HF_TOKEN_ENV_VAR),
        description="The API key to authenticate the requests to the Inference Endpoints API.",
    )

    tokenizer_id: Optional[str] = None
    model_display_name: Optional[str] = None

    structured_output: Optional[RuntimeParameter[StructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )

    _num_generations_param_supported = False

    _model_name: Optional[str] = PrivateAttr(default=None)
    _tokenizer: Optional["PreTrainedTokenizer"] = PrivateAttr(default=None)
    _api_key_env_var: str = PrivateAttr(HF_TOKEN_ENV_VAR)
    _aclient: Optional["AsyncInferenceClient"] = PrivateAttr(...)

    def load(self) -> None:  # noqa: C901
        """Loads the `AsyncInferenceClient` client to connect to the Hugging Face Inference
        Endpoint.

        Raises:
            ImportError: if the `huggingface-hub` Python client is not installed.
            ValueError: if the model is not currently deployed or is not running the TGI framework.
            ImportError: if the `transformers` Python client is not installed.
        """

        try:
            from huggingface_hub import (
                AsyncInferenceClient,
                InferenceClient,
                get_inference_endpoint,
            )
        except ImportError as ie:
            raise ImportError(
                "Hugging Face Hub Python client is not installed. Please install it using"
                " `pip install 'distilabel[hf-inference-endpoints]'`."
            ) from ie

        if self.api_key is None:
            self.api_key = SecretStr(get_hf_token(self.__class__.__name__, "api_key"))

        if self.model_id is not None:
            client = InferenceClient(
                model=self.model_id, token=self.api_key.get_secret_value()
            )
            status = client.get_model_status()

            if (
                status.state not in {"Loadable", "Loaded"}
                and status.framework != "text-generation-inference"
            ):
                raise ValueError(
                    f"Model {self.model_id} is not currently deployed or is not running the TGI framework"
                )

            self.base_url = client._resolve_url(
                model=self.model_id, task="text-generation"
            )

        if self.endpoint_name is not None:
            client = get_inference_endpoint(
                name=self.endpoint_name,
                namespace=self.endpoint_namespace,
                token=self.api_key.get_secret_value(),
            )
            if client.status in ["paused", "scaledToZero"]:
                client.resume().wait(timeout=300)
            elif client.status == "initializing":
                client.wait(timeout=300)

            self.base_url = client.url
            self._model_name = client.repository

        self._aclient = AsyncInferenceClient(
            base_url=self.base_url,
            token=self.api_key.get_secret_value(),
        )

        if self.tokenizer_id:
            try:
                from transformers import AutoTokenizer
            except ImportError as ie:
                raise ImportError(
                    "Transformers Python client is not installed. Please install it using"
                    " `pip install 'distilabel[hf-inference-endpoints]'`."
                ) from ie
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

    @property
    def model_name(self) -> Union[str, None]:  # type: ignore
        """Returns the model name used for the model."""
        return (
            self.model_display_name
            or self._model_name
            or self.model_id
            or self.endpoint_name
            or self.base_url
        )
