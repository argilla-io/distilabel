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
    inputs=[{"input": "What's 2 + 2?", "generations": "2 + 2 is 5"}]
)
print(outputs[0][0]["parsed_output"])
