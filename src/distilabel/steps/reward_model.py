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
from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import Field, PrivateAttr, SecretStr

from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.steps.base import Step, StepInput
from distilabel.utils.huggingface import HF_TOKEN_ENV_VAR

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns, StepOutput


class RewardModelScore(Step, CudaDevicePlacementMixin):
    """Assign a score to a response using a Reward Model.

    `RewardModelScore` is a `Step` that using a Reward Model (RM) loaded using `transformers`,
    assigns an score to a response generated for an instruction, or a score to a multi-turn
    conversation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        revision: if `model` refers to a Hugging Face Hub repository, then the revision
            (e.g. a branch name or a commit id) to use. Defaults to `"main"`.
        torch_dtype: the torch dtype to use for the model e.g. "float16", "float32", etc.
            Defaults to `"auto"`.
        trust_remote_code: whether to allow fetching and executing remote code fetched
            from the repository in the Hub. Defaults to `False`.
        device_map: a dictionary mapping each layer of the model to a device, or a mode like `"sequential"` or `"auto"`. Defaults to `None`.
        token: the Hugging Face Hub token that will be used to authenticate to the Hugging
            Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package
            local configuration will be used. Defaults to `None`.
        truncation: whether to truncate sequences at the maximum length. Defaults to `False`.
        max_length: maximun length to use for padding or truncation. Defaults to `None`.

    Input columns:
        - instruction (`str`, optional): the instruction used to generate a `response`.
            If provided, then `response` must be provided too.
        - response (`str`, optional): the response generated for `instruction`. If provided,
            then `instruction` must be provide too.
        - conversation (`ChatType`, optional): a multi-turn conversation. If not provided,
            then `instruction` and `response` columns must be provided.

    Output columns:
        - score (`float`): the score given by the reward model for the instruction-response
            pair or the conversation.

    Categories:
        - scorer

    Examples:
        Assigning an score for an instruction-response pair:

        ```python
        from distilabel.steps import RewardModelScore

        step = RewardModelScore(
            model="RLHFlow/ArmoRM-Llama3-8B-v0.1", device_map="auto", trust_remote_code=True
        )

        step.load()

        result = next(
            step.process(
                inputs=[
                    {
                        "instruction": "How much is 2+2?",
                        "response": "The output of 2+2 is 4",
                    },
                    {"instruction": "How much is 2+2?", "response": "4"},
                ]
            )
        )
        # [
        #   {'instruction': 'How much is 2+2?', 'response': 'The output of 2+2 is 4', 'score': 0.11690367758274078},
        #   {'instruction': 'How much is 2+2?', 'response': '4', 'score': 0.10300665348768234}
        # ]
        ```

        Assigning an score for a multi-turn conversation:

        ```python
        from distilabel.steps import RewardModelScore

        step = RewardModelScore(
            model="RLHFlow/ArmoRM-Llama3-8B-v0.1", device_map="auto", trust_remote_code=True
        )

        step.load()

        result = next(
            step.process(
                inputs=[
                    {
                        "conversation": [
                            {"role": "user", "content": "How much is 2+2?"},
                            {"role": "assistant", "content": "The output of 2+2 is 4"},
                        ],
                    },
                    {
                        "conversation": [
                            {"role": "user", "content": "How much is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                    },
                ]
            )
        )
        # [
        #   {'conversation': [{'role': 'user', 'content': 'How much is 2+2?'}, {'role': 'assistant', 'content': 'The output of 2+2 is 4'}], 'score': 0.11690367758274078},
        #   {'conversation': [{'role': 'user', 'content': 'How much is 2+2?'}, {'role': 'assistant', 'content': '4'}], 'score': 0.10300665348768234}
        # ]
        ```
    """

    model: str
    revision: str = "main"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    device_map: Union[str, Dict[str, Any], None] = None
    token: Union[SecretStr, None] = Field(
        default_factory=lambda: os.getenv(HF_TOKEN_ENV_VAR), description=""
    )
    truncation: bool = False
    max_length: Union[int, None] = None

    _model: Union["PreTrainedModel", None] = PrivateAttr(None)
    _tokenizer: Union["PreTrainedTokenizer", None] = PrivateAttr(None)

    def load(self) -> None:
        super().load()

        if self.device_map in ["cuda", "auto"]:
            CudaDevicePlacementMixin.load(self)

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "`transformers` is not installed. Please install it using `pip install transformers`."
            ) from e

        token = self.token.get_secret_value() if self.token is not None else self.token

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            token=token,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            token=token,
        )

    @property
    def inputs(self) -> "StepColumns":
        """Either `response` and `instruction`, or a `conversation` columns."""
        return {
            "response": False,
            "instruction": False,
            "conversation": False,
        }

    @property
    def outputs(self) -> "StepColumns":
        """The `score` given by the reward model."""
        return ["score"]

    def _prepare_conversation(self, input: Dict[str, Any]) -> "ChatType":
        if "instruction" in input and "response" in input:
            return [
                {"role": "user", "content": input["instruction"]},
                {"role": "assistant", "content": input["response"]},
            ]

        return input["conversation"]

    def _prepare_inputs(self, inputs: List[Dict[str, Any]]) -> "torch.Tensor":
        return self._tokenizer.apply_chat_template(  # type: ignore
            [self._prepare_conversation(input) for input in inputs],  # type: ignore
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self._model.device)  # type: ignore

    def _inference(self, inputs: List[Dict[str, Any]]) -> List[float]:
        import torch

        input_ids = self._prepare_inputs(inputs)
        with torch.no_grad():
            output = self._model(input_ids)  # type: ignore
            logits = output.logits
            if logits.shape == (2, 1):
                logits = logits.squeeze(-1)
            return logits.tolist()

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        scores = self._inference(inputs)
        for input, score in zip(inputs, scores):
            input["score"] = score
        yield inputs

    def unload(self) -> None:
        if self.device_map in ["cuda", "auto"]:
            CudaDevicePlacementMixin.unload(self)
        super().unload()
