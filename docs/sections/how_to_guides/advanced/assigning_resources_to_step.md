# Assigning resources to a `Step`

When dealing with complex pipelines that get executed in a distributed environment with abundant resources (CPUs and GPUs), sometimes it's necessary to allocate these resources judiciously among the `Step`s. This is why `distilabel` allows to specify the number of `replicas`, `cpus` and `gpus` for each `Step`. Let's see that with an example:

```python
from distilabel.pipeline import Pipeline
from distilabel.llms import vLLM
from distilabel.steps import StepResources
from distilabel.steps.tasks import PrometheusEval


with Pipeline(name="resources") as pipeline:
    ...

    prometheus = PrometheusEval(
        llm=vLLM(
            model="prometheus-eval/prometheus-7b-v2.0",
            chat_template="[INST] {{ messages[0]['content'] }}\\n{{ messages[1]['content'] }}[/INST]",
        ),
        resources=StepResources(replicas=2, cpus=1, gpus=1)
        mode="absolute",
        rubric="factual-validity",
        reference=False,
        num_generations=1,
        group_generations=False,
    )
```

In the example above, we're creating a `PrometheusEval` task (remember that `Task`s are `Step`s) that will use `vLLM` to serve `prometheus-eval/prometheus-7b-v2.0` model. This task is resource intensive as it requires an LLM, which in turn requires a GPU to run fast. With that in mind, we have specified the `resources` required for the task using the [`StepResources`][distilabel.steps.base.StepResources] class, and we have defined that we need `1` GPU and `1` CPU per replica of the task. In addition, we have defined that we need `2` replicas i.e. we will run two instances of the task so the computation for the whole dataset runs faster. In addition, `StepResources` uses the [RuntimeParametersMixin][distilabel.mixins.runtime_parameters.RuntimeParametersMixin], so we can also specify the resources for each step when running the pipeline:

```python
...

if __name__ == "__main__":
    pipeline.run(
        parameters={
            prometheus.name: {"resources": {"replicas": 2, "cpus": 1, "gpus": 1}}
        }
    )
```

And that's it! When running the pipeline, `distilabel` will create the tasks in nodes that have available the specified resources.

