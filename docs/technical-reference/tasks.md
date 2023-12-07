# Tasks

Already familiar with the `Task` component? Otherwise you can take a look at the [concepts](../concepts.md) and come back later.

## Text Generation

This tasks will allow us guiding the LLM to generate texts.

The following tasks are implemented:

- [Llama2TextGenerationTask][distilabel.tasks.text_generation.llama.Llama2TextGenerationTask]: Your go to choice to prompt your LLM for the Llama2 model.

- [OpenAITextGenerationTask][distilabel.tasks.text_generation.openai.OpenAITextGenerationTask]: The task for any chat-completion OpenAI model.

- [SelfInstructTask][distilabel.tasks.text_generation.self_instruct.SelfInstructTask]: A task following the Self-Instruct specification for building the prompts.

## Preference

- [JudgeLMTask][distilabel.tasks.preference.judgelm.JudgeLMTask]: What's this

- [UltraFeedbackTask][distilabel.tasks.preference.ultrafeedback.UltraFeedbackTask]: What's this

- [UltraJudgeTask][distilabel.tasks.preference.ultrajudge.UltraJudgeTask]: What's this
