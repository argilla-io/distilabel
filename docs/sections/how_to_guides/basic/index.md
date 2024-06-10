# Basic

`Distilabel` builds a [`Pipeline`][distilabel.pipeline.Pipeline] with steps that can be thought of as nodes in a graph, as the [`Pipeline`][distilabel.pipeline.Pipeline] will orchestrate the execution of the [`Step`][distilabel.steps.base.Step] subclasses, and those will be connected as nodes in a Direct Acyclic Graph (DAG).

This guide can be considered a tutorial, which will guide you through the different components of `distilabel`.
