from ultralabel.tasks.base import Task, get_template

_LLAMA2_TEXT_GENERATION_TEMPLATE = get_template("llama2-generation.jinja2")


class Llama2GenerationTask(Task):
    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
        " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
        " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
        " explain why instead of answering something not correct. If you don't know the answer to a"
        " question, please don't share false information."
    )

    __jinja2_template__: str = _LLAMA2_TEXT_GENERATION_TEMPLATE

    def generate_prompt(self, instruction: str) -> str:
        return self.template.render(
            system_prompt=self.system_prompt, instruction=instruction
        )

    def _parse_output(self, output: str) -> dict[str, str]:
        return {"generations": output}

    @property
    def input_args_names(self) -> list[str]:
        return ["instruction"]

    @property
    def output_args_names(self) -> list[str]:
        return ["generations"]
