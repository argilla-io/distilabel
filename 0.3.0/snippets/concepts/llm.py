from distilabel.llm import OpenAILLM
from distilabel.tasks import UltraJudgeTask

labeller = OpenAILLM(
    model="gpt-3.5-turbo",
    task=UltraJudgeTask(),
    prompt_format="openai",
    max_new_tokens=2048,
    temperature=0.0,
)

outputs = labeller.generate(
    inputs=[
        {
            "input": "Here's a math problem that you need to resolve: 2 + 2 * 3. What's the result of this problem? Explain it",
            "generations": [
                (
                    "The output of the math problem 2 + 2 * 3 is calculated by following "
                    "the order of operations (PEMDAS). First, perform the multiplication: "
                    "2 * 3 = 6. Then, perform the addition: 2 + 6 = 8. Therefore, the "
                    "output of the problem is 8."
                ),
                (
                    "The correct solution to the math problem is 8. To get the correct "
                    "answer, we follow the order of operations (PEMDAS) and perform "
                    "multiplication before addition. So, first, we solve 2 * 3 = 6, "
                    "then we add 2 to 6 to get 8."
                ),
            ],
        }
    ]
)

print(outputs[0][0]["parsed_output"])
