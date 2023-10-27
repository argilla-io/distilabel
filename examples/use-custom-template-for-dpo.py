import re
from typing import Any, Dict, List

from ultralabel.prompts import PromptTemplate


class DPOPrompt(PromptTemplate):
    chosen_criteria: str = "Is correct, accurate, and harmless."
    rejected_criteria: str = "Is incorrect, inaccurate, or harmful."

    system_prompt: str = (
        "You are a honest and reliable assistant and you've been asked to act as a"
        " labelling assistant for a set of texts."
    )
    task_description: str = (
        "Given the following set of prompt and responses, you are asked to assess whether"
        " each response is going to be chosen or rejected.\nChosen: {chosen_criteria}\n"
        "Rejected: {rejected_criteria}\n\n## Prompt\n{prompt}\n## Responses\n{responses}\n\n"
        "Now you need to decide whether each response is chosen or rejected, include not just the"
        " word `chosen` or `rejected` under each response section but also a rationale behind that"
        " decision as e.g. ## Response 1\nchosen\n...\n## Response 2\nrejected\n..., and like that"
        " for every response."
    )

    def generate_prompt(
        self, prompt: str, responses: List[str]
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.task_description.format(
                    chosen_criteria=self.chosen_criteria,
                    rejected_criteria=self.rejected_criteria,
                    prompt=prompt,
                    responses="\n".join(
                        [
                            f"{idx}. {response}"
                            for idx, response in enumerate(responses, start=1)
                        ]
                    ),
                ),
            },
        ]

    def _parse_output(self, output: str) -> List[Dict[str, Any]]:
        return [
            {"status": status, "rationale": rationale}
            for _, status, rationale in re.findall(
                r"(## Response \d+)\n{1,}(\w+)\n{0,}(.*)", output
            )
        ]

    @property
    def input_args_names(self) -> List[str]:
        return ["prompt", "responses"]

    @property
    def output_args_names(self) -> List[str]:
        return ["output"]


if __name__ == "__main__":
    from ultralabel.llm.openai_ import OpenAILLM

    prompt_template = DPOPrompt()
    llm = OpenAILLM(
        model="gpt-3.5-turbo",
        prompt_template=prompt_template,
        openai_api_key="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",
    )
    print(
        llm.generate(
            inputs=[
                {
                    "prompt": "In which continent is Spain?",
                    "responses": ["Africa", "Europe", "Oceania"],
                }
            ]
        )
    )
    # Output: [[[{'status': 'Rejected', 'rationale': 'Rationale: Spain is not located in Africa. It is located in Europe.'}, {'status': 'Chosen', 'rationale': 'Rationale: Spain is indeed located in Europe.'}, {'status': 'Rejected', 'rationale': 'Rationale: Spain is not located in Oceania. It is located in Europe.'}]]]
