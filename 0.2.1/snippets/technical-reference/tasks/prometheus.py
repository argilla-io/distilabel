from distilabel.tasks import PrometheusTask

task = PrometheusTask(
    system_prompt="You are a fair evaluator language model.",
    scoring_criteria="Relevance, Grammar, Informativeness, Engagement",
    score_descriptions={
        1: "The response is not relevant to the prompt.",
        2: "The response is relevant to the prompt, but it is not grammatical.",
        3: "The response is relevant to the prompt and it is grammatical, but it is not informative.",
        4: "The response is relevant to the prompt, it is grammatical, and it is informative, but it is not engaging.",
        5: "The response is relevant to the prompt, it is grammatical, it is informative, and it is engaging.",
    },
)
