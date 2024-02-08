import os
from distilabel.llm import OpenAILLM
from distilabel.tasks import JudgeLMTask

llm = OpenAILLM(
    task=JudgeLMTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY", None),
    temperature=0.3,
)
print(
    llm.validate_prompts(
        [
            {
                "input": "What's a large language model?",
                "generations": [
                    "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text based on vast amounts of training data.",
                    "Sorry I cannot answer that."
                ]
            }
        ]
    )[0]
)
# You are a helpful and precise assistant for checking the quality of the answer.
# [Question]
# What's an LLM?


# [The Start of Assistant 1's Answer>
# A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text based on vast amounts of training data.
# [The End of Assistant 1's Answer>
# [The Start of Assistant 2's Answer>
# Sorry I cannot answer that.
# [The End of Assistant 2's Answer>

# [System]
# We would like to request your feedback on the performance of 2 AI assistants in response to the user question displayed above.
# Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
# Please first output a single line containing only 2 values indicating the scores for Assistants 1 to 2, respectively. The 2 scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.