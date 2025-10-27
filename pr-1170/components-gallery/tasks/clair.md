---
hide:
  - navigation
---
# CLAIR

Contrastive Learning from AI Revisions (CLAIR).



CLAIR uses an AI system to minimally revise a solution A→A´ such that the resulting
    preference A `preferred` A’ is much more contrastive and precise.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[task]
			ICOL1[student_solution]
		end
		subgraph New columns
			OCOL0[revision]
			OCOL1[rational]
			OCOL2[model_name]
		end
	end

	subgraph CLAIR
		StepInput[Input Columns: task, student_solution]
		StepOutput[Output Columns: revision, rational, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **task** (`str`): The task or instruction.

- **student_solution** (`str`): An answer to the task that is to be revised.




#### Outputs


- **revision** (`str`): The revised text.

- **rational** (`str`): The rational for the provided revision.

- **model_name** (`str`): The name of the model used to generate the revision and rational.





### Examples


#### Create contrastive preference pairs
```python
from distilabel.steps.tasks import CLAIR
from distilabel.models import InferenceEndpointsLLM

llm=InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 4096,
    },
)
clair_task = CLAIR(llm=llm)

clair_task.load()

result = next(
    clair_task.process(
        [
            {
                "task": "How many gaps are there between the earth and the moon?",
                "student_solution": 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.'
            }
        ]
    )
)
# result
# [{'task': 'How many gaps are there between the earth and the moon?',
# 'student_solution': 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.',
# 'revision': 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
# 'rational': 'The student\'s solution provides a clear and concise answer to the question. However, there are a few areas where it can be improved. Firstly, the term "gaps" can be misleading in this context. The student should clarify what they mean by "gaps." Secondly, the student provides some additional information about the Moon\'s orbit, which is correct but could be more clearly connected to the main point. Lastly, the student\'s conclusion could be more concise.',
# 'distilabel_metadata': {'raw_output_c_l_a_i_r_0': '{teacher_reasoning}: The student\'s solution provides a clear and concise answer to the question. However, there are a few areas where it can be improved. Firstly, the term "gaps" can be misleading in this context. The student should clarify what they mean by "gaps." Secondly, the student provides some additional information about the Moon\'s orbit, which is correct but could be more clearly connected to the main point. Lastly, the student\'s conclusion could be more concise.\n\n{corrected_student_solution}: There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
# 'raw_input_c_l_a_i_r_0': [{'role': 'system',
#     'content': "You are a teacher and your task is to minimally improve a student's answer. I will give you a {task} and a {student_solution}. Your job is to revise the {student_solution} such that it is clearer, more correct, and more engaging. Copy all non-corrected parts of the student's answer. Do not allude to the {corrected_student_solution} being a revision or a correction in your final solution."},
#     {'role': 'user',
#     'content': '{task}: How many gaps are there between the earth and the moon?\n\n{student_solution}: There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.\n\n-----------------\n\nLet\'s first think step by step with a {teacher_reasoning} to decide how to improve the {student_solution}, then give the {corrected_student_solution}. Mention the {teacher_reasoning} and {corrected_student_solution} identifiers to structure your answer.'}]},
# 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
```




### References

- [Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment](https://arxiv.org/abs/2408.06266v1)

- [APO and CLAIR - GitHub Repository](https://github.com/ContextualAI/CLAIR_and_APO)


