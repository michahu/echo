import argparse
import os
import random
import json
import math

def main():
    parser = argparse.ArgumentParser(description="Generate a random order of experiment configurations.")
    parser.add_argument("--n", type=int, required=True, help="Number of unique experiments to generate.")
    parser.add_argument("--length", required=True, help="Number of config files to sample for each experiment, or 'max' to use all.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the configuration files.")
    args = parser.parse_args()

    random.seed(0)

    config_files = []
    for f in os.listdir(args.config_dir):
        if not f.endswith(".json"):
            continue
        file_path = os.path.join(args.config_dir, f)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as config_file:
                    json.load(config_file)
                config_files.append(f)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid json file {f}")

    if args.length == "max":
        length = len(config_files)
    else:
        length = int(args.length)

    if len(config_files) < length:
        raise ValueError("The number of config files is less than the length of an experiment.")

    num_possible_permutations = math.perm(len(config_files), length)
    if args.n > num_possible_permutations:
        raise ValueError(f"Cannot generate {args.n} unique experiments. Only {num_possible_permutations} are possible.")

    experiments = set()
    # Loop until we have n unique experiments.
    while len(experiments) < args.n:
        experiment = tuple(random.sample(config_files, length))
        experiments.add(experiment)

    output_file_path = os.path.join(args.config_dir, "experiment_order.txt")
    with open(output_file_path, "w") as f:
        for exp in experiments:
            full_path_exp = [os.path.join(args.config_dir, file) for file in exp]
            f.write(" ".join(full_path_exp) + "\n")

if __name__ == "__main__":
    main()
