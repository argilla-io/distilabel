# Using a file system to pass data of batches between steps

In some situations, it can happen that the batches contains so much data that is faster to write it to disk and read it back in the next step, instead of passing it using the queue. To solve this issue, `distilabel` uses [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) to allow providing a file system configuration and whether if this file system should be used to pass data between steps in the `run` method of the `distilabel` pipelines:

```python
from distilabel.pipeline import Pipeline

with Pipeline(name="my-pipeline") as pipeline:
  ...

if __name__ == "__main__":
    distiset = pipeline.run(
        ..., 
        storage_parameters={"protocol": "gcs", "path": "gcs://my-bucket"},
        use_fs_to_pass_data=True
    )
```

The code above setups a file system (in this case Google Cloud Storage) and sets the flag `use_fs_to_pass_data` to specify that the data of the batches should be passed to the steps using the file system.The `storage_parameters` argument is optional, and in the case it's not provided but `use_fs_to_pass_data==True`, `distilabel` will use the local file system.

!!! NOTE

    As `GlobalStep`s receives all the data from the previous steps in one single batch accumulating all the data, it's very likely that the data of the batch will be too big to be passed using the queue. In this case and even if `use_fs_to_pass_data==False`, `distilabel` will use the file system to pass the data to the `GlobalStep`. 

