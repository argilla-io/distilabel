# Scaling and distributing a pipeline with Ray

Although the local [Pipeline][distilabel.pipeline.local.Pipeline] based on [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) + [serving LLMs with an external service](serving_an_llm_for_reuse.md) is enough for executing most of the pipelines used to create SFT and preference datasets, there are scenarios where you might need to scale your pipeline across multiple machines. In such cases, distilabel leverages [Ray](https://www.ray.io/) to distribute the workload efficiently. This allows you to generate larger datasets, reduce execution time, and maximize resource utilization across a cluster of machines, without needing to change a single line of code.

## Relation between distilabel steps and Ray Actors

A `distilabel` pipeline consist of several [`Step`][distilabel.steps.base.Step]s. An `Step` is a class that defines a basic life-cycle:

1. It will load or create the resources (LLMs, clients, etc) required to run its logic.
2. It will run a loop waiting for incoming batches received using a queue. Once it receives one batch, it will process it and put the processed batch into an output queue.
3. When it finish a batch that is the final one or receives a special signal, the loop will finish and the unload logic will be executed.

So an `Step` needs to maintain a minimum state and the best way to do that with Ray is using [actors](https://docs.ray.io/en/latest/ray-core/actors.html).

``` mermaid
graph TD
    A[Step] -->|has| B[Multiple Replicas]
    B -->|wrapped in| C[Ray Actor]
    C -->|maintains| D[Step Replica State]
    C -->|executes| E[Step Lifecycle]
    E -->|1. Load/Create Resources| F[LLMs, Clients, etc.]
    E -->|2. Process batches from| G[Input Queue]
    E -->|3. Processed batches are put in| H[Output Queue]
    E -->|4. Unload| I[Cleanup]

```

## Executing a pipeline with Ray

The recommended way to execute a `distilabel` pipeline using Ray is using the [Ray Jobs API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html#ray-jobs-api).

Before jumping on the explanation, let's first install the prerequisites:
```bash
pip install distilabel[ray]
```

!!! tip

    It's recommended to create a virtual environment.

For the purpose of explaining how to execute a pipeline with Ray, we'll use the following pipeline throughout the examples:

```python
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration

with Pipeline(name="text-generation-ray-pipeline") as pipeline:
    load_data_from_hub = LoadDataFromHub(output_mappings={"prompt": "instruction"})

    text_generation = TextGeneration(
        llm=vLLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        )
    )

    load_data_from_hub >> text_generation

if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            load_data_from_hub.name: {
                "repo_id": "HuggingFaceH4/instruction-dataset",
                "split": "test",
            },
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 4096,
                    }
                },
                "resources": {"replicas": 2, "gpus": 1}, # (1)
            },
        }
    )

    distiset.push_to_hub(
        "<YOUR_HF_USERNAME_OR_ORGANIZATION>/text-generation-distilabel-ray" # (2)
    )
```

1. We're setting [resources](assigning_resources_to_step.md) for the `text_generation` step and defining that we want two replicas and one GPU per replica. `distilabel` will create two replicas of the step i.e. two actors in the Ray cluster, and each actor will request to be allocated in a node of the cluster that have at least one GPU. You can read more about how Ray manages the resources [here](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#resources).
2. You should modify this and add your user or organization on the Hugging Face Hub.

It's a basic pipeline with just two steps: one to load a dataset from the Hub with an `instruction` column and one to generate a `response` for that instruction using Llama 3 8B Instruct with [vLLM](/distilabel/components-gallery/llms/vllm/). Simple but enough to demonstrate how to distribute and scale the workload using a Ray cluster!

### Using Ray Jobs API

If you don't know the Ray Jobs API then it's recommended to read [Ray Jobs Overview](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html#ray-jobs-overview). Quick summary: Ray Jobs is the recommended way to execute a job in a Ray cluster as it will handle packaging, deploying and managing the Ray application.

To execute the pipeline above, we first need to create a directory (kind of a package) with the pipeline script (or scripts)
that we will submit to the Ray cluster:

```bash
mkdir ray-pipeline
```

The content of the directory `ray-pipeline` should be:

```
ray-pipeline/
├── pipeline.py
└── runtime_env.yaml
```

The first file contains the code of the pipeline, while the second one (`runtime_env.yaml`) is a specific Ray file containing the [environment dependencies](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#environment-dependencies) required to run the job:

```yaml
pip:
  - distilabel[ray,vllm] >= 1.3.0
env_vars:
  HF_TOKEN: <YOUR_HF_TOKEN>
```

With this file we're basically informing to the Ray cluster that it will have to install `distilabel` with the `vllm` and `ray` extra dependencies to be able to run the job. In addition, we're defining the `HF_TOKEN` environment variable that will be used (by the `push_to_hub` method) to upload the resulting dataset to the Hugging Face Hub. 

After that, we can proceed to execute the `ray` command that will submit the job to the Ray cluster:
```bash
ray job submit \
    --address http://localhost:8265 \
    --working-dir ray-pipeline \
    --runtime-env ray-pipeline/runtime_env.yaml -- python pipeline.py
```

What this will do, it's to basically upload the `--working-dir` to the Ray cluster, install the dependencies and then execute the `python pipeline.py` command from the head node.

## File system requirements

As described in [Using a file system to pass data to steps](fs_to_pass_data.md), `distilabel` relies on the file system to pass the data to the `GlobalStep`s, so if the pipeline to be executed in the Ray cluster have any `GlobalStep` or do you want to set the `use_fs_to_pass_data=True` of the [run][distilabel.pipeline.local.Pipeline.run] method, then you will need to setup a file system to which all the nodes of the Ray cluster have access:

```python
if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={...},
        storage_parameters={"path": "file:///mnt/data"}, # (1)
        use_fs_to_pass_data=True,
    )
```

1. All the nodes of the Ray cluster should have access to `/mnt/data`.

## Executing a `RayPipeline` in a cluster with Slurm

If you have access to an HPC, then you're probably also a user of [Slurm](https://slurm.schedmd.com/), a workload manager typically used on HPCs. We can create Slurm job that takes some nodes and deploy a Ray cluster to run a distributed `distilabel` pipeline:

```bash
#!/bin/bash
#SBATCH --job-name=distilabel-ray-text-generation
#SBATCH --partition=your-partition
#SBATCH --qos=normal
#SBATCH --nodes=2 # (1)
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1 # (2)
#SBATCH --gpus-per-node=1 # (3)
#SBATCH --time=0:30:00

set -ex

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Activate virtual environment
source /path/to/virtualenv/.venv/bin/activate

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Generate a unique Ray tmp dir for the head node (just in case the default one is not writable)
head_tmp_dir="/tmp/ray_tmp_${SLURM_JOB_ID}_head"

echo "Starting HEAD at $head_node"
OUTLINES_CACHE_DIR="/tmp/.outlines" srun --nodes=1 --ntasks=1 -w "$head_node" \ # (4)
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir="$head_tmp_dir" \
    --block &

# Give some time to head node to start...
echo "Waiting a bit before starting worker nodes..."
sleep 10

# Start Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))

# Start from 1 (0 is head node)
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    worker_tmp_dir="/tmp/ray_tmp_${SLURM_JOB_ID}_worker_$i"
    echo "Starting WORKER $i at $node_i"
    OUTLINES_CACHE_DIR="/tmp/.outlines" srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --temp-dir="$worker_tmp_dir" \
        --block &
    sleep 5
done

# Give some time to the Ray cluster to gather info
echo "Waiting a bit before submitting the job..."
sleep 60

# Finally submit the job to the cluster
ray job submit --address http://localhost:8265 --working-dir ray-pipeline -- python -u pipeline.py
```

1. In this case, we just want two nodes: one to run the Ray head node and one to run a worker.
2. We just want to run a task per node i.e. the Ray command that starts the head/worker node.
3. We have selected 1 GPU per node, but we could have selected more depending on the pipeline.
4. We need to set the environment variable `OUTLINES_CACHE_DIR` to `/tmp/.outlines` to avoid issues with the nodes trying to read/write the same `outlines` cache files, which is not possible.

## `vLLM` and `tensor_parallel_size`

In order to use `vLLM` multi-GPU and multi-node capabilities with `ray`, we need to do a few changes in the example pipeline from above. The first change needed is to specify a value for `tensor_parallel_size` aka "In how many GPUs do I want you to load the model", and the second one is to define `ray` as the `distributed_executor_backend` as the default one in `vLLM` is to use `multiprocessing`:


```python
with Pipeline(name="text-generation-ray-pipeline") as pipeline:
    load_data_from_hub = LoadDataFromHub(output_mappings={"prompt": "instruction"})

    text_generation = TextGeneration(
        llm=vLLM(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer="meta-llama/Meta-Llama-3.1-70B-Instruct",
            extra_kwargs={
                "tensor_parallel_size": 8,
                "distributed_executor_backend": "ray",
            }
        )
    )

    load_data_from_hub >> text_generation
```

More information about distributed inference with `vLLM` can be found here: [vLLM - Distributed Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
