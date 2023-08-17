import json

from modal import Retries
from .common import (
    stub,
    VOL_MOUNT_PATH,
    output_vol,
    get_data_path
)

@stub.function(
    retries=Retries(
        max_retries=3,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
    timeout=60 * 60 * 2,
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp",
)
def load_data_sql(data_dir: str = "data_sql"):
    from datasets import load_dataset

    dataset = load_dataset("b-mc2/sql-create-context")

    dataset_splits = {"train": dataset["train"]}
    out_path = get_data_path(data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for key, ds in dataset_splits.items():
        with open(out_path, "w") as f:
            for item in ds:
                newitem = {
                    "input": item["question"],
                    "context": item["context"],
                    "output": item["answer"],
                }
                f.write(json.dumps(newitem) + "\n")
    