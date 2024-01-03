from distilabel.tasks import TextGenerationTask, Task
from distilabel.llm import ProcessLLM, LLM


def load_gpt_4(task: Task) -> LLM:
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-4",
        task=task,
        num_threads=4,
    )


llm = ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_gpt_4)
future = llm.generate(
    inputs=[{"input": "Write a letter for Bob"}], num_generations=1
)  # (1)
llm.teardown()  # (2)
result = future.result()
# >>> print(result[0][0]["parsed_output"]["generations"])
# Dear Bob,
# I hope this letter finds you in good health and high spirits. I know it's been a while since we last caught up, and I wanted to take the time to connect and share a few updates.
# Life has been keeping me pretty busy lately. [Provide a brief overview of what you've been up to: work, school, family, hobbies, etc.]
# I've often found myself reminiscing about the good old days, like when we [include a memorable moment or shared experience with Bob].
