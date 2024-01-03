from datasets import Dataset

xkcd_instructions = Dataset.from_dict(
    {"input": ["Could you imagine an interview process going sideways?"]}
)
ds_xkcd = pipe_full.generate(xkcd_instructions, num_generations=3)
