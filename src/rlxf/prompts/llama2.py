from typing import Any, List


class Llama2Prompt:
    @staticmethod
    def chat_format(instruction: str, *args: Any, **kwargs: Any) -> str:
        system_prompt: str = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
            " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
            " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
            " explain why instead of answering something not correct. If you don't know the answer to a"
            " question, please don't share false information."
        )
        return "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]".format(
            system_prompt=system_prompt, instruction=instruction
        )

    @staticmethod
    def rank_format(prompt: str, responses: List[str]) -> str:
        system_prompt: str = (
            "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible,"
            " and do not introduce any bias or toxicity into your responses."
        )
        instruction: str = (
            "You are going to be provided with a list of responses to a prompt. Please rank the responses"
            " from most helpful to least helpful according to the given prompt. For example, if you are given N"
            " responses, you should rank from most helpful to least helpful response, with no additional explanation,"
            " just the number of the position of the response in the list. The following output shows how the ranking"
            " should look like assuming 1 is the most helpful/accurate and N is the worst: 1>2>...>N \n\n"
            f" The prompt was {prompt}, and the generated respones were: {responses}"
        )
        return "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST] The ranking is: ".format(
            system_prompt=system_prompt, instruction=instruction
        )
