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
