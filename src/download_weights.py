"""Download weights."""

from modal import Image, Stub

from .common import (
    stub, output_vol, VOL_MOUNT_PATH, user_data_path, user_model_path
)
import os
import json
from pathlib import Path

@stub.function(
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp"
)
def load_model(user: str):
    """Load model."""
    path = user_model_path(user)
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
def main(user: str, output_dir: str):
    # copy adapter_config.json and adapter_model.bin files into dict
    load_model.call(user=user)
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
    
    
    
