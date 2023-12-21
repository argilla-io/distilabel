from distilabel.tasks import TextGenerationTask, Task
from distilabel.llm import ProcessLLM, LLM, LLMPool

def load_gpt_3(task: Task) -> LLM:
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-3.5-turbo",
        task=task,
        num_threads=4,
    )

def load_gpt_4(task: Task) -> LLM:
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-4",
        task=task,
        num_threads=4,
    )


pool = LLMPool(llms=[
    ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_gpt_3),
    ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_gpt_4),
])
result = pool.generate(
    inputs=[{"input": "Write a letter for Bob"}], num_generations=2
)
pool.teardown()
# >>> print(result[0][0]["parsed_output"]["generations"], end="\n\n\n\n\n\n---->")
# Dear Bob,
# I hope this letter finds you in good health and high spirits. I know it's been a while since we last caught up, and I wanted to take the time to connect and share a few updates.
# Life has been keeping me pretty busy lately. [Provide a brief overview of what you've been up to: work, school, family, hobbies, etc.]
# I've often found myself reminiscing about the good old days, like when we [include a memorable moment or shared experience with Bob].
# >>> print(result[0][1]["parsed_output"]["generations"])
# Of course, I'd be happy to draft a sample letter for you. However, I would need some additional 
# information including who "Bob" is, the subject matter of the letter, the tone (formal or informal), 
# and any specific details or points you'd like to include. Please provide some more context and I'll do my best to assist you.
