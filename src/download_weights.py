"""Download weights."""

from .common import (
    stub, output_vol, VOL_MOUNT_PATH, get_model_path
)
import os
import json
from pathlib import Path

@stub.function(
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp"
)
def load_model(model_dir: str = "data_sql"):
    """Load model."""
    path = get_model_path(model_dir=model_dir)
    config_path = path / "adapter_config.json"
    model_path = path / "adapter_model.bin"

    config_data = json.load(open(config_path))
    with open(model_path, "rb") as f:
        model_data = f.read()

        print(f'loaded config, model data from {path}')

        # read data, put this in `model_dict` on stub
        stub.model_dict["config"] = config_data
        stub.model_dict["model"] = model_data

@stub.local_entrypoint()
def main(output_dir: str, model_dir: str = "data_sql"):
    # copy adapter_config.json and adapter_model.bin files into dict
    load_model.call(model_dir=model_dir)
    model_data = stub.model_dict["model"]
    config_data = stub.model_dict["config"]

    print(f"Loaded model data, storing in {output_dir}")
    
    # store locally
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out_model_path = Path(output_dir) / "adapter_model.bin"
    out_config_path = Path(output_dir) / "adapter_config.json"

    with open(out_model_path, "wb") as f:
        f.write(model_data)
    
    with open(out_config_path, "w") as f:
        json.dump(config_data, f)
    
    print("Done!")
    
    
    
