from distilabel.llm import TogetherInferenceLLM
from distilabel.tasks import TextGenerationTask

llm = TogetherInferenceLLM(
    model="togethercomputer/llama-2-70b-chat",
    task=TextGenerationTask(),
    max_new_tokens=512,
    temperature=0.3,
    prompt_format="llama2",
)
output = llm.generate(
    [{"input": "Explain me the theory of relativity as if you were a pirate."}]
)
# >>> print(result[0][0]["parsed_output"]["generations"])
# Ahoy matey! Yer lookin' fer a tale of the theory of relativity, eh? Well,
# settle yerself down with a pint o' grog and listen close, for this be a story
# of the sea of time and space!
# Ye see, matey, the theory of relativity be tellin' us that time and space ain't
# fixed things, like the deck o' a ship or the stars in the sky. Nay, they be like
# the ocean itself, always changin' and flowin' like the tides.
# Now, imagine ...
