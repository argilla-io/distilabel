from typing import Dict, List

from rlxf.prompts import PromptTemplate
from typing_extensions import TypedDict


class Label(TypedDict):
    label: str
    label_description: str


class TextClassificationPrompt(PromptTemplate):
    labels: List[Label]

    __type__: str = "text-classification"

    system_prompt: str = (
        "You are a honest and reliable assistant and you've been asked to act as a"
        " labelling assistant for a set of texts."
    )
    task_description: str = (
        "Given the following set of finstruction and labels, you are asked to provide"
        " the correct label or labels in the form of [LABELS]a,b,c[/LABELS]. Also the"
        " existing labels are only and just only the following:\n{labels}\n\n[TEXT]"
        "{text}[/TEXT]"
    )

    def generate_prompt(self, text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.task_description.format(
                    labels="\n".join(
                        [
                            f"* {label['label']}: {label['label_description']}"
                            for label in self.labels
                        ]
                    ),
                    text=text,
                ),
            },
            {"role": "assistant", "content": "[LABELS]"},
        ]

    def parse_output(self, output: str) -> List[str]:
        return output.split("[/LABELS]")[0].split(",")

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction"]

    @property
    def output_args_names(self) -> List[str]:
        return ["labels"]


if __name__ == "__main__":
    from rlxf.llm.openai_ import OpenAILLM

    prompt_template = TextClassificationPrompt(
        labels=[
            Label(label="Sports", label_description="Related to any sport."),
            Label(
                label="Politics", label_description="Related to politics in general."
            ),
            Label(
                label="News",
                label_description="Just a text in the format of a newspaper new.",
            ),
        ]
    )

    llm = OpenAILLM(
        model="gpt-4",
        prompt_template=prompt_template,
        openai_api_key="<YOUR_OPENAI_API_KEY>",
    )
    print(llm.generate(inputs=[{"text": "Messi scored 3 goals yesterday!"}]))
    # Output: [["Sports"]]
