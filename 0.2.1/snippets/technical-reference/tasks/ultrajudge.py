from distilabel.tasks import UltraJudgeTask

# To see the complete system_prompt and task_description please take a look at the UltraJudgeTask definition
ultrajudge_task = UltraJudgeTask(
    system_prompt="You are an evaluator tasked with assessing AI assistants' responses from the perspective of typical user preferences...",
    task_description="Your task is to rigorously evaluate the performance of...",
    areas=[
        "Practical Accuracy",
        "Clarity & Transparency",
        "Authenticity & Reliability",
        "Compliance with Intent",
    ],
)
