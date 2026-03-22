import glob
import os

base_path = "./data/different_types"
dir_names = glob.glob("experiments/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    os.system(f"python scripts/shims/inference_warp.py --base_path {base_path} --case_name {case_name}")
