import yaml
import argparse
from pprint import pprint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fp_1",type=str,required=True)
    parser.add_argument("--config_fp_2",type=str,required=True)
    args = parser.parse_args()

    with open(args.config_fp_1,"r") as f:
        config_d_1 = yaml.load(f,Loader=yaml.FullLoader)

    with open(args.config_fp_2,"r") as f:
        config_d_2 = yaml.load(f,Loader=yaml.FullLoader)

    all_keys = set(config_d_1.keys()) | set(config_d_2.keys())

    miss_d_1 = set()
    miss_d_2 = set()
    diff_d_1 = {}
    diff_d_2 = {}

    for key in all_keys:
        if key not in config_d_1:
            miss_d_1.add(key)
        elif key not in config_d_2:
            miss_d_2.add(key)
        elif config_d_1[key] != config_d_2[key]:
            diff_d_1[key] = config_d_1[key]
            diff_d_2[key] = config_d_2[key]
    
    for key in diff_d_1.keys():
        print(f"> key = {key}")
        print(diff_d_1[key])
        print(diff_d_2[key])
        print()

    print("> missing in 1")
    pprint(miss_d_1)
    print()

    print("> missing in 2")
    pprint(miss_d_2)
    print()
