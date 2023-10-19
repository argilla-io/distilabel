from typing import Any, List
from textwrap import dedent

from rlxf.prompts.base import Prompt


class GPT4Prompt(Prompt):
    @staticmethod
    def chat_format(instruction: str, *args: Any, **kwargs: Any) -> str:
        pass

    @staticmethod
    def rank_format(prompt: str, responses: List[str]) -> str:
        system_prompt: str = (
            "You are a helpful, respectful, and honest assistant. Your role is to evaluate"
            " text quality based on given criteria."
        )
        instruction: str = dedent("""
            # Informativeness / Helpfulness Assessment

            Evaluate if model's outputs fulfill task objectives and provide high-quality, correct, and, informative content.

            Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativeness.

            **Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.

            Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
            1. **Severely Incorrect**: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.
            2. **Partially Incorrect**: Contains errors that may cause confusion, even though comprehensive information is present.
            3. **Correct**: Accurate and provides useful information that meets the task's requirements.
            4. **Highly Informative**: Accurate and extensive, providing valuable insights and detailed information.
            5. **Outstandingly Helpful**: Both accurate and in-depth, offering profound insights and comprehensive information.

            ---

            ## Format

            ### Input
            Instruction: [Specify task goal and restrictions]

            Texts:
            {text_sections_input}

            ### Output
            {text_sections_output}

            ---

            ## Annotation

            ### Input
            Instruction: {{instruction}}

            Texts:
            {{text_sections_annotation}}

            ### Output
        """)
        formatted_instruction = instruction.format(
            text_sections_input="\n".join(f"<text {i + 1}> [Text {i + 1}]" for i in range(len(responses))),
            text_sections_output="\n\n".join(
                dedent(
                    f"""
                    #### Output for Text {i + 1}
                    Rating: [Rating for text {i + 1}]
                    Rationale: [Rationale for the rating in short sentences]
                    """
                )
                for i in range(len(responses))
            ),
        )
        return f"{system_prompt}\n\n{formatted_instruction}"
