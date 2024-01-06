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

import re

prompt_default_format = re.compile(
    r"(?P<system_prompt>.+)\n(?P<formatted_prompt>.+)", re.MULTILINE
)

prompt_llama2_format = re.compile(
    r"<s>\[INST] <<SYS>>\n(?P<system_prompt>.+)<<\/SYS>>\n\n(?P<formatted_prompt>.+) \[\/INST]"
)

prompt_chatml_format = re.compile(
    r"<\|im_start\|>system\n(?P<system_prompt>.+)<\|im_end\|>\n<\|im_start\|>user\n(?P<formatted_prompt>.+)<\|im_end\|>\n<\|im_start\|>assistant\n"
)

prompt_zephyr_format = re.compile(
    r"<\|system\|>\n(?P<system_prompt>.+)</s>\n<\|user\|>\n(?P<formatted_prompt>.+)</s>\n<\|assistant\|>\n"
)
