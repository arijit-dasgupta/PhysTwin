import glob
import json
import os

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    os.system(
        f"python scripts/shims/train_warp.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
    )
